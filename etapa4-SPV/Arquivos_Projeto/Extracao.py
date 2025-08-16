import cv2
import mediapipe as mp
import csv
import os

# Inicializa MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao abrir a câmera")
    exit()

# Nome do gesto (pode alterar para capturar gestos diferentes)
nome_gesto = input("Digite o nome do gesto que será coletado: ").strip()

# Nome do arquivo CSV para salvar os dados
csv_filename = 'dados_gesto.csv'

# Se o arquivo não existir, cria com cabeçalho
if not os.path.isfile(csv_filename):
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header += [f'x{i+1}', f'y{i+1}', f'z{i+1}']
        header.append('label')
        writer.writerow(header)

print(f"Comece a coletar dados para o gesto: {nome_gesto}")
print("Pressione 's' para salvar o frame atual, 'q' para sair.")

# Cria uma única janela
cv2.namedWindow("Detecção de Mãos", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar o frame.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Detecção de Mãos', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # Salvar dados do frame atual
        if results.multi_hand_landmarks:
            # Apenas uma mão, primeira detectada
            hand = results.multi_hand_landmarks[0]
            coords = []
            for lm in hand.landmark:
                coords += [lm.x, lm.y, lm.z]

            coords.append(nome_gesto)

            # Salvar no CSV
            with open(csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(coords)

            print(f"Dado salvo para o gesto '{nome_gesto}'")
        else:
            print("Nenhuma mão detectada para salvar.")

    elif key == ord('q'):
        print("Encerrando coleta de dados.")
        break

cap.release()
cv2.destroyAllWindows()
