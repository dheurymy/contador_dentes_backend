from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import math

app = FastAPI()

# CORS para permitir frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega o modelo YOLO
model = YOLO("backend/yolo/best.pt")

@app.post("/contar_dentes")
async def contar_dentes(file: UploadFile = File(...)):
    # Lê a imagem enviada
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Roda o modelo
    results = model(img)
    boxes = results[0].boxes.xyxy  # coordenadas das caixas

    # Primeiro, calcular o centro da engrenagem (média dos centros dos dentes)
    centros = []
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centros.append((cx, cy))

    eng_cx = int(sum([c[0] for c in centros]) / len(centros))
    eng_cy = int(sum([c[1] for c in centros]) / len(centros))

    # Criar lista com ângulos para ordenar
    dentes = []
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        angle = math.atan2(cy - eng_cy, cx - eng_cx)
        dentes.append({
            "cx": cx,
            "cy": cy,
            "angle": angle
        })

    # Ordenar dentes pelo ângulo (sentido horário)
    dentes.sort(key=lambda d: d["angle"])

    # Desenhar pontos e números radialmente
    for i, dente in enumerate(dentes, start=1):
        cx = int(dente["cx"])
        cy = int(dente["cy"])
        angle = dente["angle"]

        # ponto vermelho
        cv2.circle(img, (cx, cy), radius=4, color=(0, 0, 255), thickness=-1)

        # distância radial para afastar o número
        distancia = 35

        # posição do número
        num_x = int(cx + math.cos(angle) * distancia)
        num_y = int(cy + math.sin(angle) * distancia)

        # --- FUNDO BRANCO ATRÁS DO NÚMERO ---
        texto = str(i)
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        escala = 0.7
        espessura = 2

        # tamanho do texto
        (w, h), _ = cv2.getTextSize(texto, fonte, escala, espessura)

        # coordenadas do retângulo branco
        rect_x1 = num_x - 4
        rect_y1 = num_y - h - 4
        rect_x2 = num_x + w + 4
        rect_y2 = num_y + 4

        # desenha fundo branco
        cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)

        # desenha contorno preto (opcional, mas fica lindo)
        cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), 1)

        # desenha número por cima
        cv2.putText(
            img,
            texto,
            (num_x, num_y),
            fonte,
            escala,
            (0, 0, 255),
            espessura
        )

    # Converte imagem para base64
    _, buffer = cv2.imencode(".jpg", img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "dentes_detectados": len(dentes),
        "imagem_processada": img_base64
    }
