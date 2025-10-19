import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

# --- CONFIGURACIÓN ---
# Directorio base del proyecto
basedir = os.path.abspath(os.path.dirname(__file__))

# Configuración de carpetas
UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
SPECTROGRAM_FOLDER = os.path.join(basedir, 'spectrograms')

# Asegurarse de que las carpetas existan
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SPECTROGRAM_FOLDER, exist_ok=True)

# Inicialización de la App Flask
app = Flask(__name__)

# Configuración de la Base de Datos (SQLite)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SPECTROGRAM_FOLDER'] = SPECTROGRAM_FOLDER

# Inicialización de la Base de Datos
db = SQLAlchemy(app)

# --- MODELO DE BASE DE DATOS ---
class Grabacion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nombre_archivo = db.Column(db.String(300), nullable=False)
    path_archivo = db.Column(db.String(500), nullable=False)
    path_espectrograma = db.Column(db.String(500), nullable=False)
    
    # --- Metadatos de la investigación ---
    locacion = db.Column(db.String(200), nullable=True)
    investigador = db.Column(db.String(100), nullable=True)
    descripcion = db.Column(db.Text, nullable=True)
    tags = db.Column(db.String(300), nullable=True)

    def __repr__(self):
        return f'<Grabacion {self.nombre_archivo}>'

# --- INICIALIZAR LA BASE DE DATOS ---
@app.cli.command('init-db')
def init_db_command():
    """Crea las tablas de la base de datos."""
    db.create_all()
    print('Base de datos inicializada.')

# --- RUTAS DE LA APLICACIÓN ---

@app.route('/')
def index():
    """Página principal: Muestra todas las grabaciones."""
    lista_grabaciones = Grabacion.query.all()
    return render_template('index.html', grabaciones=lista_grabaciones)


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Página para subir archivos."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(audio_path)
            
            # --- PROCESAMIENTO DE AUDIO (Librosa) ---
            spectrogram_filename = f"{os.path.splitext(filename)[0]}.png"
            spectrogram_path = os.path.join(app.config['SPECTROGRAM_FOLDER'], spectrogram_filename)
            
            try:
                y, sr = librosa.load(audio_path)
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
                S_db = librosa.power_to_db(S, ref=np.max)
                
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
                plt.colorbar(format='%+2.0f dB')
                plt.title('Mel-frequency spectrogram')
                plt.tight_layout()
                plt.savefig(spectrogram_path)
                plt.close()
                print(f"Espectrograma guardado en: {spectrogram_path}")

            except Exception as e:
                print(f"Error procesando el audio: {e}")
                spectrogram_filename = "error.png"
            
            # --- Guardar en Base de Datos ---
            nueva_grabacion = Grabacion(
                nombre_archivo=filename,
                path_archivo=audio_path,
                path_espectrograma=spectrogram_filename,
                locacion=request.form.get('locacion'),
                investigador=request.form.get('investigador'),
                descripcion=request.form.get('descripcion'),
                tags=request.form.get('tags')
            )
            
            db.session.add(nueva_grabacion)
            db.session.commit()
            
            return redirect(url_for('index'))

    return render_template('upload.html')


# --- RUTAS ESPECIALES PARA SERVIR ARCHIVOS ---

@app.route('/play/<filename>')
def play_audio(filename):
    """Sirve un archivo de audio desde la carpeta 'uploads'."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/spectrogram/<filename>')
def get_spectrogram(filename):
    """Sirve una imagen de espectrograma desde la carpeta 'spectrograms'."""
    return send_from_directory(app.config['SPECTROGRAM_FOLDER'], filename)

# --- Punto de entrada para correr la app ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') # host='0.0.0.0' es importante para Codespaces