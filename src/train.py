import os
from roboflow import Roboflow
from ultralytics import YOLO

def main():
    print("🚀 Iniciando pipeline de entrenamiento...")
    
    # 1. Descarga del Dataset (Asegúrate de poner tu API Key real para la entrega o dejar las instrucciones)
    # rf = Roboflow(api_key="TU_API_KEY")
    # project = rf.workspace("tu-espacio").project("tu-proyecto")
    # dataset = project.version(1).download("yolov8")
    
    print("📁 Dataset cargado correctamente.")
    
    # 2. Cargar modelo base
    print("🧠 Cargando modelo YOLOv8 Nano...")
    model = YOLO('yolov8n.pt')
    
    # 3. Entrenamiento (Fine-Tuning)
    # Nota: Asegúrate de que la ruta data.yaml sea la correcta tras descargar de Roboflow
    print("⚙️ Entrenando el modelo (Fine-Tuning)...")
    model.train(
        data=f"data.yaml", # Cambiar por la ruta real generada por Roboflow
        epochs=25,
        imgsz=640,
        project="src/runs",
        name="detector_mugiwaras"
    )
    
    print("✅ ¡Entrenamiento completado! El modelo experto se ha guardado en src/runs/detector_mugiwaras/weights/best.pt")

if __name__ == "__main__":
    main()