import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import tempfile
import cv2
import os
import pandas as pd
import numpy as np

# ─── Configuración de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Detección de Residuos Reciclables",
    page_icon="♻️",
    layout="wide",
)

BASE_DIR    = Path(__file__).parent
MODEL_PATH  = BASE_DIR / "models" / "best.pt"

@st.cache_resource
def load_model(path: str):
    return YOLO(path)

model       = load_model(str(MODEL_PATH))
CLASS_NAMES = model.names

# ─── Cabecera ──────────────────────────────────────────────────────────────────
st.title("♻️ Detección de Residuos Reciclables")
st.markdown(
    """
    **Proyecto 2 – Visión Computacional con Deep Learning**  
    Modelo: `YOLO11s` &nbsp;|&nbsp; Dataset: `residuosdataset`  
    Clases: `Electronics · Glass · Metal · Miscellaneous · Organic · Paper · Plastic`
    """
)
st.divider()

# ─── Barra lateral ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parámetros de inferencia")
    conf_threshold = st.slider("Umbral de confianza", 0.10, 0.95, 0.40, 0.05)
    iou_threshold  = st.slider("Umbral de IoU (NMS)",  0.10, 0.95, 0.45, 0.05)
    st.markdown("---")
    st.markdown("**Clases detectables:**")
    for idx, name in CLASS_NAMES.items():
        st.markdown(f"- `{idx}` → {name}")

# ─── Carga del video ───────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📁 Carga tu video de prueba",
    type=["mp4", "avi", "mov", "mkv"],
    help="Formatos soportados: mp4, avi, mov, mkv",
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(uploaded_file.read())
        input_path = tmp_in.name

    st.info(f"Archivo cargado: **{uploaded_file.name}** — {uploaded_file.size / 1024:.1f} KB")

    if st.button("🚀 Ejecutar detección", use_container_width=True, type="primary"):

        cap    = cv2.VideoCapture(input_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        st.subheader("🎬 Detección en tiempo real")

        # Placeholder para mostrar cada frame anotado
        frame_placeholder = st.empty()
        progress_bar      = st.progress(0, text="Procesando…")

        detections_log = []
        frame_idx      = 0

        # Buffer para guardar frames anotados (para descarga posterior)
        annotated_frames = []
        frame_size       = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results   = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
            annotated = results[0].plot()  # BGR numpy array con bounding boxes

            # Guardar para el video de descarga
            annotated_frames.append(annotated)
            if frame_size is None:
                frame_size = (annotated.shape[1], annotated.shape[0])

            # Convertir BGR → RGB para mostrarlo con st.image
            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            # Log de detecciones
            for box in results[0].boxes:
                detections_log.append({
                    "Frame":     frame_idx,
                    "Clase":     CLASS_NAMES[int(box.cls.item())],
                    "Confianza": round(box.conf.item(), 3),
                })

            frame_idx += 1
            if total > 0:
                progress_bar.progress(
                    min(frame_idx / total, 1.0),
                    text=f"Frame {frame_idx} / {total}",
                )

        cap.release()
        progress_bar.empty()
        st.success(f"✅ Procesamiento completado — {frame_idx} fotogramas analizados")

        # ─── Generar video para descarga ──────────────────────────────────────
        if annotated_frames and frame_size:
            output_path = input_path.replace(".mp4", "_detected.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"avc1")   # H.264 nativo en OpenCV
            writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

            # Si avc1 no funciona en el sistema, fallback a mp4v
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

            for f in annotated_frames:
                writer.write(f)
            writer.release()

            with open(output_path, "rb") as f:
                st.download_button(
                    label="⬇️ Descargar video procesado",
                    data=f,
                    file_name="residuos_detectados.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

        # ─── Estadísticas ─────────────────────────────────────────────────────
        if detections_log:
            st.subheader("📊 Estadísticas de detección")
            df = pd.DataFrame(detections_log)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total detecciones",  len(df))
            col2.metric("Clases detectadas",  df["Clase"].nunique())
            col3.metric("Confianza promedio", f"{df['Confianza'].mean():.3f}")

            st.markdown("**Detecciones por clase:**")
            st.bar_chart(df["Clase"].value_counts())

            with st.expander("📋 Ver tabla completa"):
                st.dataframe(df, use_container_width=True)
        else:
            st.warning("⚠️ No se detectaron objetos. Prueba bajando el umbral de confianza.")

        # Limpieza
        if os.path.exists(input_path):
            os.unlink(input_path)

else:
    st.markdown(
        """
        <div style="text-align:center; padding:60px;
                    background-color:#f0f2f6; border-radius:12px;">
            <h3>👆 Sube un video para comenzar</h3>
            <p>El modelo analizará cada fotograma y mostrará las detecciones en tiempo real.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
