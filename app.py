import os
import io
import pickle
import base64
import numpy as np
from flask import Flask, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from skimage.filters.rank import median
from skimage.morphology import disk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration
data_dir = 'uploads'
ALLOWED_EXTENSIONS = {'mat'}

# Ensure upload directory exists
os.makedirs(data_dir, exist_ok=True)

# Helper to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper to encode a Matplotlib figure to base64 string
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = data_dir

# Buffer for last pickle
buf_pk = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    global buf_pk
    # Save uploaded .mat files
    feat = request.files.get('feature_file')
    lbl  = request.files.get('label_file')
    if not feat or not lbl or not allowed_file(feat.filename) or not allowed_file(lbl.filename):
        return "Invalid files", 400
    feat_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(feat.filename))
    lbl_path  = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(lbl.filename))
    feat.save(feat_path)
    lbl.save(lbl_path)

    # Load data
    cube = loadmat(feat_path)['indian_pines_corrected']
    gt   = loadmat(lbl_path)['indian_pines_gt']

    # Reshape and mask
    H, W, B = cube.shape
    X = cube.reshape(-1, B)
    y = gt.reshape(-1)
    mask = y > 0
    X_masked = X[mask]
    y_masked = y[mask]

    # Standard scale + train/test split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_masked)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_masked, test_size=0.3, random_state=42, stratify=y_masked
    )

    # Denoise cube
    cube_sp = np.empty_like(cube)
    for b in range(B): cube_sp[:,:,b] = gaussian_filter(cube[:,:,b], sigma=1)
    cube_dn = np.empty_like(cube_sp)
    for i in range(H):
        for j in range(W):
            cube_dn[i,j,:] = savgol_filter(cube_sp[i,j,:], window_length=7, polyorder=3, mode='mirror')
    X_dn = cube_dn.reshape(-1, B)[mask]

    # PCA + Logistic Regression
    X_pca = PCA(n_components=30).fit_transform(X_dn)
    Xp_scaled = StandardScaler().fit_transform(X_pca)
    clf = LogisticRegression(solver='saga', max_iter=3000).fit(Xp_scaled, y_masked)
    y_pred = clf.predict(Xp_scaled)
    acc = accuracy_score(y_masked, y_pred)
    pca_cm = confusion_matrix(y_masked, y_pred)
    # pca_cm = pca_cm.astype('float') / pca_cm.sum(axis=1)[:, np.newaxis]
    pca_report = classification_report(y_masked, y_pred)

    # LDA raw vs denoised
    y_dn_trial = gt.reshape(-1)[mask]
    print("X den shape:", X_dn.shape)
    print("y_masked shape:", y_dn_trial.shape)
    X_tr, X_te, y_tr, y_te = train_test_split(X_dn, y_dn_trial, test_size=0.3,
                                          random_state=42, stratify=y_dn_trial)
    X_raw = X_masked
    lda_raw = Pipeline([('scale', StandardScaler()), ('lda', LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'))]).fit(X_raw, y_masked)
    lda_den = Pipeline([('scale', StandardScaler()), ('lda', LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'))]).fit(X_tr, y_tr)
    pred_raw = lda_raw.predict(X_raw)
    pred_den = lda_den.predict(X_dn)
    raw_map = np.zeros(mask.shape, int); raw_map[mask]=pred_raw; raw_map=raw_map.reshape(H,W)
    den_map = np.zeros(mask.shape, int); den_map[mask]=pred_den; den_map=den_map.reshape(H,W)
    den_map_sm = median(den_map.astype(np.uint8), disk(2))
    lda_report = classification_report(y_dn_trial, pred_den)
    raw_acc = accuracy_score(y_masked, pred_raw)
    den_acc = accuracy_score(y_masked, pred_den)
    den_sm_acc = accuracy_score(y_masked, den_map_sm.flatten()[mask])

    # Generate PCA confusion matrix plot
    # fig1, ax1 = plt.subplots()
    # ConfusionMatrixDisplay(pca_cm).plot(ax=ax1)
    # ax1.set_title(f"PCA+LogReg Acc={acc:.3f}")
    # img1 = fig_to_base64(fig1)
    pca_map = np.zeros(mask.shape, int); pca_map[mask]=y_pred; pca_map=raw_map.reshape(H,W)
    fig2, axs2 = plt.subplots(1,3,figsize=(15,5))
    axs2[0].imshow(gt, cmap='tab20'); axs2[0].set_title('GT'); axs2[0].axis('off')
    axs2[1].imshow(raw_map, cmap='tab20'); axs2[1].set_title(f"Denoised PCA Acc={acc:.3f}"); axs2[1].axis('off')
    # axs2[2].imshow(den_map_sm, cmap='tab20'); axs2[2].set_title(f"Den+LDA Acc={den_sm_acc:.3f}"); axs2[2].axis('off')
    axs2[2].text(0, 1, pca_report, fontsize=10, verticalalignment='top', family='monospace')

    # ConfusionMatrixDisplay(pca_cm).plot(ax=axs2[2])

    plt.tight_layout()
    img1 = fig_to_base64(fig2)

    # Generate LDA comparison plot
    fig2, axs2 = plt.subplots(1,3,figsize=(15,5))
    axs2[0].imshow(gt, cmap='tab20'); axs2[0].set_title('GT'); axs2[0].axis('off')
    # axs2[1].imshow(raw_map, cmap='tab20'); axs2[1].set_title(f"Raw LDA Acc={raw_acc:.3f}"); axs2[1].axis('off')
    axs2[1].imshow(den_map_sm, cmap='tab20'); axs2[2].set_title(f"Den+LDA Acc={den_sm_acc:.3f}"); axs2[2].axis('off')
    axs2[2].text(0, 1, lda_report, fontsize=10, verticalalignment='top', family='monospace')
    plt.tight_layout()
    img2 = fig_to_base64(fig2)

    # Pack pickle
    results = {
        'pca': {'acc': acc, 'conf_matrix': pca_cm},
        'lda': {'raw_acc': raw_acc, 'den_sm_acc': den_sm_acc}
    }
    buf_pk = io.BytesIO()
    pickle.dump(results, buf_pk)
    buf_pk.seek(0)

    return render_template('results.html', img_pca=img1, img_lda=img2, download_url=url_for('download'))

@app.route('/download')
def download():
    return send_file(buf_pk, download_name='results.pkl', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
