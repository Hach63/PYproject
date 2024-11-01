from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from flask_mail import Mail, Message
import os
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier



app = Flask(__name__)

# Configuration de Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'hachmibarhoumi52@gmail.com'  # Remplacez par votre adresse e-mail
app.config['MAIL_PASSWORD'] = 'cocg wnzt jqlh htem'  # Remplacez par votre mot de passe
app.config['MAIL_DEFAULT_SENDER'] = 'vdk@gmail.com'
app.secret_key = 'PWD'

UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Exemfrom sklearn.neighbors import KNeighborsClassifier

import cv2
import numpy as np

# Fonction pour calculer l'histogramme LBP
def calculer_lbp(image):
    hauteur, largeur = image.shape
    lbp_image = np.zeros((hauteur, largeur), dtype=np.uint8)

    for i in range(1, hauteur - 1):
        for j in range(1, largeur - 1):
            centre = image[i, j]
            code = 0
            code |= (image[i - 1, j - 1] > centre) << 7
            code |= (image[i - 1, j] > centre) << 6
            code |= (image[i - 1, j + 1] > centre) << 5
            code |= (image[i, j + 1] > centre) << 4
            code |= (image[i + 1, j + 1] > centre) << 3
            code |= (image[i + 1, j] > centre) << 2
            code |= (image[i + 1, j - 1] > centre) << 1
            code |= (image[i, j - 1] > centre)
            lbp_image[i, j] = code

    histogram, _ = np.histogram(lbp_image.ravel(), bins=256, range=(0, 256))
    histogram = histogram.astype("float32")  # Conversion en float32
    histogram /= (histogram.sum() + 1e-6)  # Normalisation
    return histogram

# Modèles LBP pour chaque catégorie de texture (exemple complet avec des valeurs fictives)
# Exemple de modèles LBP basés sur des histogrammes réels
categories_textures = {
    "T-shirt"=="Tissu en coton doux": calculer_lbp(cv2.imread("static/images/coton_doux.jpg", cv2.IMREAD_GRAYSCALE)),
    "Denim robuste": calculer_lbp(cv2.imread("static/images/denim.jpg", cv2.IMREAD_GRAYSCALE)),
    "Cuir lisse": calculer_lbp(cv2.imread("static/images/cuir.jpg", cv2.IMREAD_GRAYSCALE)),
    "Soie fluide": calculer_lbp(cv2.imread("static/images/soie.jpg", cv2.IMREAD_GRAYSCALE)),
    "Feutre rigide": calculer_lbp(cv2.imread("static/images/feutre.jpg", cv2.IMREAD_GRAYSCALE)),
}

# Fonction pour prédire la texture de l'image
def obtenir_description_texture(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Impossible de charger l'image à partir de {image_path}")
    
    histogram_lbp = calculer_lbp(image)
    
    scores = {categorie: cv2.compareHist(histogram_lbp, hist_ref, cv2.HISTCMP_CORREL)
              for categorie, hist_ref in categories_textures.items()}
    
    return max(scores, key=scores.get)

# Liste des produits avec chemin d'accès `static/images`
produits = [
    {"nom": "T-shirt", "prix": 20.00, "image": "static/images/tshirt.jpg", "categorie": "homme"},
    {"nom": "Jean", "prix": 40.00, "image": "static/images/jean.jpg", "categorie": "homme"},
    {"nom": "Chaussures", "prix": 60.00, "image": "static/images/chaussures.jpg", "categorie": "homme"},
    {"nom": "Robe", "prix": 50.00, "image": "static/images/robe.jpg", "categorie": "femme"},
    {"nom": "Chapeaux", "prix": 40.00, "image": "static/images/chapeaux.jpg", "categorie": "femme"},
    {"nom": "Chaussures", "prix": 60.00, "image": "static/images/chaussures1.jpg", "categorie": "femme"},
    {"nom": "Jean Enfant", "prix": 40.00, "image": "static/images/kids_jean.jpg", "categorie": "enfant"},
    {"nom": "Chemise Enfant", "prix": 10.00, "image": "static/images/kids_shirt.jpg", "categorie": "enfant"},
    {"nom": "Ensemble Enfant", "prix": 50.00, "image": "static/images/kids_ensemble.jpg", "categorie": "enfant"},
    {"nom": "Chaussures Enfant", "prix": 30.00, "image": "static/images/kids_chaussures.jpg", "categorie": "enfant"},
]

# Utiliser LBP pour classifier la texture et ajouter une description de texture pour chaque produit
for produit in produits:
    chemin_image = produit["image"]
    produit["description_texture"] = obtenir_description_texture(chemin_image)

# Affichage des produits avec leur description de texture
for produit in produits:
    print(f"{produit['nom']} - {produit['description_texture']}")



# Mise à jour des produits avec analyse automatique de texture



mail = Mail(app)

# Fonction pour extraire les caractéristiques d'une image
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    if descriptors is None:
        return np.array([])
    return descriptors.flatten()

# Fonction pour trouver des images similaires
# Fonction pour trouver des images similaires
def find_similar_images(uploaded_image_path):
    uploaded_features = extract_features(uploaded_image_path)
    
    if uploaded_features.size == 0:
        return []  # Aucun feature détecté dans l'image uploadée

    image_dir = app.config['UPLOAD_FOLDER']
    uploaded_image_name = os.path.basename(uploaded_image_path).split('.')[0]
    
    # Collecter toutes les caractéristiques et noms d'images
    features_list = []
    image_names = []
    
    for image_name in os.listdir(image_dir):
        if uploaded_image_name in image_name:
            image_path = os.path.join(image_dir, image_name)
            image_features = extract_features(image_path)
            
            if image_features.size > 0:
                features_list.append(image_features)
                image_names.append(image_name)
    
    if not features_list:
        return []
    
    # Normaliser la taille des features
    min_length = min(len(f) for f in features_list)
    features_array = np.array([f[:min_length] for f in features_list])
    uploaded_features = uploaded_features[:min_length].reshape(1, -1)
    
    # Initialiser et ajuster le modèle KNN
    n_neighbors = min(5, len(features_list))  # Nombre de voisins à trouver
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(features_array)
    
    # Trouver les images les plus similaires
    distances, indices = knn.kneighbors(uploaded_features)
    
    # Convertir les distances en scores de similarité (1 - distance)
    similarities = [(image_names[idx], float(1 - dist)) for idx, dist in zip(indices[0], distances[0])]
    
    return similarities


# Route pour la page d'index avec fonctionnalité d'upload
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Créer le dossier si nécessaire avant de sauvegarder le fichier
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            # Trouver des images similaires
            similar_images = find_similar_images(file_path)
            return render_template('index.html', uploaded_image=file.filename, similar_images=similar_images)
    
    return render_template('index.html')

# Route pour la page des produits
@app.route('/produit')
def produit():
    return render_template('produits.html', produits=produits)

# Routes pour gérer le panier
@app.route('/ajouter_au_panier', methods=['POST'])
def ajouter_au_panier():
    data = request.get_json()
    produit = {
        'nom': data['nom'],
        'prix': data['prix'],
        'image': data['image']
    }

    if 'panier' not in session:
        session['panier'] = []

    session['panier'].append(produit)
    session.modified = True

    return jsonify({'success': True, 'cart_count': len(session['panier'])})

@app.route('/panier', methods=['GET', 'POST'])
def panier():
    if request.method == 'POST':
        # Récupération des coordonnées du client depuis le formulaire
        client_name = request.form.get('client_name')
        client_email = request.form.get('client_email')
        client_phone = request.form.get('client_phone')

        # Enregistrement des coordonnées dans la session
        session['client_info'] = {
            'name': client_name,
            'email': client_email,
            'phone': client_phone
        }
        flash('Coordonnées du client enregistrées avec succès !', 'success')

    panier_articles = session.get('panier', [])
    client_info = session.get('client_info', {})

    return render_template('panier.html', articles=panier_articles, client_info=client_info)

@app.route('/retirer_du_panier', methods=['POST'])
def retirer_du_panier():
    data = request.get_json()
    nom = data['nom']

    if 'panier' in session:
        session['panier'] = [produit for produit in session['panier'] if produit['nom'] != nom]

    return jsonify({'success': True, 'cart_count': len(session['panier'])})

# Route pour la page de contact avec envoi d'e-mail
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        if not name or not email or not message:
            flash("Tous les champs doivent être remplis!", 'danger')
            return redirect(url_for('contact'))

        try:
            msg = Message(subject=f'Nouveau message de {name}',
                          recipients=['hachmibarhoumi52@gmail.com'],
                          body=f'Nom: {name}\nEmail: {email}\n\nMessage:\n{message}')
            
            mail.send(msg)
            flash('Votre message a été envoyé avec succès!', 'success')
        except Exception as e:
            flash(f"Une erreur s'est produite lors de l'envoi de l'e-mail : {str(e)}", 'danger')
        
        return redirect(url_for('contact'))
    
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
    