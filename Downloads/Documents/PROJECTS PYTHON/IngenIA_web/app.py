from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_mail import Mail, Message
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuración del servidor de correo
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('EMAIL_USER')
app.config['MAIL_PASSWORD'] = os.getenv('EMAIL_PASSWORD')

mail = Mail(app)

# Servir archivos estáticos
@app.route('/')
def index():
    return send_from_directory('frontend/public', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('frontend/public', filename)

# Servir archivos CSS
@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory('frontend/public/css', filename)

# Servir archivos JS
@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('frontend/public/js', filename)

# Servir imágenes
@app.route('/images/<path:filename>')
def serve_images(filename):
    return send_from_directory('frontend/public/images', filename)

@app.route('/api/contact', methods=['POST'])
def contact():
    try:
        data = request.json
        msg = Message('Nuevo mensaje de contacto',
                    sender=app.config['MAIL_USERNAME'],
                    recipients=['e.morales@uni.pe'])
        
        body = f"""
        Nuevo mensaje de contacto:
        
        Nombre: {data.get('nombre')}
        Email: {data.get('email')}
        Teléfono: {data.get('telefono')}
        Asunto: {data.get('asunto')}
        Mensaje: {data.get('mensaje')}
        """
        
        msg.body = body
        mail.send(msg)
        
        return jsonify({"message": "Mensaje enviado correctamente"}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/presupuesto', methods=['POST'])
def presupuesto():
    try:
        data = request.json
        msg = Message('Nueva solicitud de presupuesto',
                    sender=app.config['MAIL_USERNAME'],
                    recipients=['e.morales@uni.pe'])
        
        body = f"""
        Nueva solicitud de presupuesto:
        
        Nombre: {data.get('nombre')}
        Email: {data.get('email')}
        Teléfono: {data.get('telefono')}
        Servicio: {data.get('servicio')}
        Mensaje: {data.get('mensaje')}
        """
        
        msg.body = body
        mail.send(msg)
        
        return jsonify({"message": "Solicitud enviada correctamente"}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port) 