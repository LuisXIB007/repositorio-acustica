import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import or_
from werkzeug.utils import secure_filename
import uuid 

# --- CONFIGURACIÓN ---
# Directorio base del proyecto
basedir = os.path.abspath(os.path.dirname(__file__))

# Inicialización de la App Flask
app = Flask(__name__)

# AÑADE ESTA LÍNEA (50 * 1024 * 1024 = 50MB)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 


# Configuración de carpetas
UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
SPECTROGRAM_FOLDER = os.path.join(basedir, 'spectrograms')

SPECTROGRAM_FOLDER = os.path.join(basedir, 'spectrograms')
os.makedirs(SPECTROGRAM_FOLDER, exist_ok=True)
app.config['SPECTROGRAM_FOLDER'] = SPECTROGRAM_FOLDER

# Asegurarse de que las carpetas existan
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SPECTROGRAM_FOLDER, exist_ok=True)



# Configuración de la Base de Datos (SQLite)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SPECTROGRAM_FOLDER'] = SPECTROGRAM_FOLDER

# Inicialización de la Base de Datos
db = SQLAlchemy(app)

# --- MODELO DE BASE DE DATOS ---
class Aula(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    edificio = db.Column(db.String(100), nullable=False)
    nombre_aula = db.Column(db.String(100), nullable=False)
    num_ventanas = db.Column(db.Integer, nullable=True)
    superficies = db.relationship('Superficie', backref='aula', lazy=True, cascade="all, delete-orphan")
    
    # Conexión 1: Un Aula puede tener MUCHAS Grabaciones
    grabaciones = db.relationship('Grabacion', backref='aula', lazy=True)
    
    imagenes = db.relationship('ImagenAula', backref='aula', lazy=True)

    def __repr__(self):
        return f'<Aula {self.edificio} - {self.nombre_aula}>'

class Superficie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nombre_espacio = db.Column(db.String(200), nullable=False)
    material = db.Column(db.String(200), nullable=False)
    area = db.Column(db.Float, nullable=False)

    # Vínculo: Esta superficie pertenece a UNA Aula
    aula_id = db.Column(db.Integer, db.ForeignKey('aula.id'), nullable=False)

    def __repr__(self):
        return f'<Superficie {self.nombre_espacio}: {self.material}>'

class ImagenAula(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(300), nullable=False) # Nombre del archivo guardado
    
    # Vínculo: Esta imagen pertenece a UNA Aula
    aula_id = db.Column(db.Integer, db.ForeignKey('aula.id'), nullable=False)

    def __repr__(self):
        return f'<Imagen {self.filename}>'

class Grabacion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nombre_archivo = db.Column(db.String(300), nullable=False)
    path_archivo = db.Column(db.String(500), nullable=False)
    path_espectrograma = db.Column(db.String(500), nullable=False)
    
    # ... metadatos ...
    investigador = db.Column(db.String(100), nullable=True)
    descripcion = db.Column(db.Text, nullable=True)
    
    # Vínculo: Esta grabación pertenece a UNA Aula
    aula_id = db.Column(db.Integer, db.ForeignKey('aula.id'), nullable=False)

    def __repr__(self):
        return f'<Grabacion {self.nombre_archivo}>'


# --- INICIALIZAR LA BASE DE DATOS ---
@app.cli.command('init-db')
def init_db_command():
    """Crea las tablas de la base de datos."""
    db.create_all()
    print('Base de datos inicializada.')


# --- RUTAS DE LA APLICACIÓN ---
# Definir la carpeta de imágenes de aulas
AULA_IMG_FOLDER = os.path.join(app.config['UPLOAD_FOLDER'], 'aulas_img')
os.makedirs(AULA_IMG_FOLDER, exist_ok=True)


@app.route('/')
def index():
    """Página principal: Muestra todas las AULAS, con filtro de búsqueda."""

    # 1. Obtener el término de búsqueda de la URL (ej: /?q=termino)
    query = request.args.get('q')

    if query:
        # 2. Si hay búsqueda, filtrar la base de datos
        # Preparamos el término para buscar "dentro" de las palabras
        search_term = f"%{query}%" 

        # 3. Buscamos en múltiples columnas
        # .ilike() ignora mayúsculas/minúsculas
        lista_aulas = Aula.query.filter(
            or_(
                Aula.edificio.ilike(search_term),
                Aula.nombre_aula.ilike(search_term),
                Aula.materiales.ilike(search_term)
            )
        ).all()

    else:
        # 4. Si NO hay búsqueda, simplemente mostrar todo
        lista_aulas = Aula.query.all()

    # 5. Renderizar la misma plantilla, pero con la lista filtrada
    return render_template('index.html', aulas=lista_aulas)

@app.route('/aula/<int:aula_id>')
def aula_detalle(aula_id):
    # 1. Buscar en la BD el aula con ese ID.
    # Usamos .get_or_404() que es un atajo genial:
    # O encuentra el aula, o muestra un error "404 No Encontrado"
    aula = Aula.query.get_or_404(aula_id)

    # 2. Renderizar un *nuevo* template y pasarle el aula encontrada
    return render_template('aula_detalle.html', aula=aula)

@app.route('/aulas/add', methods=['GET', 'POST'])
def add_aula():
    """Página para registrar una nueva aula."""
    if request.method == 'POST':
        # 1. Crear el objeto Aula con los datos del formulario
        nueva_aula = Aula(
            edificio=request.form.get('edificio'),
            nombre_aula=request.form.get('nombre_aula'),
            num_ventanas=int(request.form.get('num_ventanas', 0))
        )
        
        # 2. Añadirla a la sesión para obtener su ID
        db.session.add(nueva_aula)
        db.session.flush() # 'flush' asigna el ID sin guardar permanentemente

        nombres_espacio = request.form.getlist('nombre_espacio[]')
        materiales = request.form.getlist('material[]')
        areas = request.form.getlist('area[]')

        for nombre, mat, area in zip(nombres_espacio, materiales, areas):
        # Solo guardar si el usuario llenó los 3 campos de la fila
            if nombre and mat and area:
                nueva_superficie = Superficie(
                    nombre_espacio=nombre,
                    material=mat,
                    area=float(area.replace(',', '.')), # Reemplaza coma por punto
                    aula_id=nueva_aula.id
                )
                db.session.add(nueva_superficie)

        # 3. Procesar y guardar las imágenes
        imagenes = request.files.getlist('imagenes')
        for img in imagenes:
            if img.filename != '':
                # Crear un nombre de archivo único
                filename = secure_filename(img.filename)
                ext = os.path.splitext(filename)[1]
                unique_filename = f"{uuid.uuid4()}{ext}"
                
                # Guardar el archivo físicamente
                img_path = os.path.join(AULA_IMG_FOLDER, unique_filename)
                img.save(img_path)
                
                # Crear el registro en la BD para la imagen
                nueva_imagen = ImagenAula(
                    filename=unique_filename,
                    aula_id=nueva_aula.id
                )
                db.session.add(nueva_imagen)

        # 4. Guardar todo (el aula y sus imágenes) en la BD
        db.session.commit()
        
        return redirect(url_for('index')) # Redirigir al dashboard

    # Si el método es GET, solo muestra el formulario
    return render_template('add_aula.html')


@app.route('/uploads/aulas_img/<filename>')
def get_aula_image(filename):
    """Sirve las imágenes de las aulas."""
    return send_from_directory(AULA_IMG_FOLDER, filename)


# --- RUTAS ESPECIALES PARA SERVIR ARCHIVOS ---

# --- Punto de entrada para correr la app ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') # host='0.0.0.0' es importante para Codespaces
    
@app.route('/aula/<int:aula_id>/edit', methods=['GET', 'POST'])
def edit_aula(aula_id):
    # 1. Obtener el aula que queremos editar
    aula = Aula.query.get_or_404(aula_id)
    
    # 2. Si el usuario envía el formulario (POST)
    if request.method == 'POST':
        # --- Actualizar datos del aula ---
        aula.edificio = request.form.get('edificio')
        aula.nombre_aula = request.form.get('nombre_aula')
        aula.num_ventanas = int(request.form.get('num_ventanas', 0))

        # --- NUEVO: Borrar superficies seleccionadas ---
        ids_a_borrar = request.form.getlist('delete_superficie')
        for s_id in ids_a_borrar:
            superficie_a_borrar = Superficie.query.get(s_id)
            if superficie_a_borrar:
                db.session.delete(superficie_a_borrar)

        # --- NUEVO: Añadir nuevas superficies ---
        nombres_espacio = request.form.getlist('nombre_espacio[]')
        materiales = request.form.getlist('material[]')
        areas = request.form.getlist('area[]')

        for nombre, mat, area in zip(nombres_espacio, materiales, areas):
            if nombre and mat and area:
                nueva_superficie = Superficie(
                    nombre_espacio=nombre,
                    material=mat,
                    area=float(area.replace(',', '.')),
                    aula_id=aula.id
                )
                db.session.add(nueva_superficie)
    # --- FIN DE LAS NUEVAS SECCIONES ---
        
        # --- Borrar imágenes seleccionadas ---
        imagenes_a_borrar_ids = request.form.getlist('delete_image')
        for img_id in imagenes_a_borrar_ids:
            img_a_borrar = ImagenAula.query.get(img_id)
            if img_a_borrar:
                # Borrar el archivo físico del disco
                try:
                    os.remove(os.path.join(AULA_IMG_FOLDER, img_a_borrar.filename))
                    print(f"Archivo borrado: {img_a_borrar.filename}")
                except OSError as e:
                    print(f"Error borrando archivo {img_a_borrar.filename}: {e}")
                
                # Borrar el registro de la base de datos
                db.session.delete(img_a_borrar)
        
        # --- Añadir nuevas imágenes ---
        # (Este código es idéntico al de 'add_aula')
        imagenes = request.files.getlist('imagenes')
        for img in imagenes:
            if img.filename != '':
                # Crear un nombre de archivo único
                filename = secure_filename(img.filename)
                ext = os.path.splitext(filename)[1]
                unique_filename = f"{uuid.uuid4()}{ext}"
                
                # Guardar el archivo físicamente
                img_path = os.path.join(AULA_IMG_FOLDER, unique_filename)
                img.save(img_path)
                
                # Crear el registro en la BD para la imagen
                nueva_imagen = ImagenAula(
                    filename=unique_filename,
                    aula_id=aula.id
                )
                db.session.add(nueva_imagen)

        # --- Guardar todos los cambios en la BD ---
        db.session.commit()
        
        # Redirigir de vuelta a la página de detalles
        return redirect(url_for('aula_detalle', aula_id=aula.id))
    
    # 3. Si el usuario solo está cargando la página (GET)
    # Simplemente mostramos el formulario pre-llenado
    return render_template('edit_aula.html', aula=aula)

@app.route('/aula/<int:aula_id>/delete', methods=['POST'])
def delete_aula(aula_id):
    # 1. Encontrar el aula
    aula = Aula.query.get_or_404(aula_id)

    # 2. Borrar todos los archivos de IMAGEN asociados
    for img in aula.imagenes:
        try:
            # Borrar el archivo físico del disco
            os.remove(os.path.join(AULA_IMG_FOLDER, img.filename))
        except OSError as e:
            print(f"Error borrando archivo de imagen {img.filename}: {e}")

        # Borrar el registro de la imagen de la BD
        db.session.delete(img)

    # 3. Borrar todos los archivos de GRABACIÓN asociados
    for grabacion in aula.grabaciones:
        try:
            # Borrar el archivo de audio (este ya tiene la ruta completa)
            os.remove(grabacion.path_archivo) 
            # Borrar el archivo de espectrograma
            os.remove(os.path.join(app.config['SPECTROGRAM_FOLDER'], grabacion.path_espectrograma))
        except OSError as e:
            print(f"Error borrando archivos de grabación {grabacion.nombre_archivo}: {e}")

        # Borrar el registro de la grabación de la BD
        db.session.delete(grabacion)

    # 4. Ahora que los "hijos" están borrados, borrar el "padre" (el Aula)
    db.session.delete(aula)

    # 5. Guardar todos los cambios en la BD
    db.session.commit()

    # 6. Redirigir al usuario al dashboard principal
    return redirect(url_for('index'))

@app.route('/play/<filename>')
def play_audio(filename):
    """Sirve un archivo de audio desde la carpeta 'uploads'."""
    # Nota: UPLOAD_FOLDER ya debería estar definida
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/spectrogram/<filename>')
def get_spectrogram(filename):
    """Sirve una imagen de espectrograma desde la carpeta 'spectrograms'."""
    return send_from_directory(app.config['SPECTROGRAM_FOLDER'], filename)

@app.route('/aula/<int:aula_id>/upload_audio', methods=['POST'])
def upload_audio(aula_id):
    # 1. Verificar que el aula exista (aunque el POST solo viene de ahí)
    aula = Aula.query.get_or_404(aula_id)

    # 2. Verificar que el archivo exista en el formulario
    if 'file' not in request.files:
        print('No se encontró el archivo')
        return redirect(request.url) # Recargar la página

    file = request.files['file']

    if file.filename == '':
        print('Nombre de archivo vacío')
        return redirect(request.url)

    if file:
        # 3. Guardar el archivo de audio original
        filename = secure_filename(file.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(audio_path)

        # 4. PROCESAMIENTO DE AUDIO (¡La parte de Librosa!)
        spectrogram_filename = f"{os.path.splitext(filename)[0]}_{uuid.uuid4()}.png"
        spectrogram_path = os.path.join(app.config['SPECTROGRAM_FOLDER'], spectrogram_filename)

        try:
            # Cargar el audio
            y, sr = librosa.load(audio_path, sr=None) # sr=None preserva la frec. de muestreo original

            # Generar Mel Spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr/2) # fmax hasta Nyquist
            S_db = librosa.power_to_db(S, ref=np.max)

            # Crear la imagen con matplotlib
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', fmax=sr/2)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel-frequency spectrogram')
            plt.tight_layout()
            plt.savefig(spectrogram_path)
            plt.close() # MUY importante para liberar memoria
            print(f"Espectrograma guardado en: {spectrogram_path}")

        except Exception as e:
            print(f"Error procesando el audio: {e}")
            spectrogram_filename = "error.png" # Deberías tener una imagen 'error.png'

        # 5. Obtener metadatos del formulario
        investigador = request.form.get('investigador')
        descripcion = request.form.get('descripcion')

        # 6. Guardar la grabación en la Base de Datos
        nueva_grabacion = Grabacion(
            nombre_archivo=filename,
            path_archivo=audio_path, # Guardamos la ruta completa
            path_espectrograma=spectrogram_filename, # Solo el nombre del .png
            investigador=investigador,
            descripcion=descripcion,
            aula_id=aula.id  # ¡El vínculo clave!
        )

        db.session.add(nueva_grabacion)
        db.session.commit()

        # 7. Redirigir de vuelta a la página de detalles del aula
        return redirect(url_for('aula_detalle', aula_id=aula.id))

    return redirect(url_for('aula_detalle', aula_id=aula.id))

@app.route('/grabacion/<int:grabacion_id>/delete', methods=['POST'])
def delete_audio(grabacion_id):
    # 1. Encontrar la grabación
    grabacion = Grabacion.query.get_or_404(grabacion_id)

    # Guardamos el ID del aula ANTES de borrar, para saber a dónde volver
    aula_id_redirect = grabacion.aula_id

    # 2. Borrar los archivos físicos
    try:
        os.remove(grabacion.path_archivo) # Borrar el .wav/.mp3
        os.remove(os.path.join(app.config['SPECTROGRAM_FOLDER'], grabacion.path_espectrograma)) # Borrar el .png
    except OSError as e:
        print(f"Error borrando archivos de grabación {grabacion.nombre_archivo}: {e}")

    # 3. Borrar el registro de la base de datos
    db.session.delete(grabacion)
    db.session.commit()

    # 4. Redirigir de vuelta a la página de detalles del aula
    return redirect(url_for('aula_detalle', aula_id=aula_id_redirect))