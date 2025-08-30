import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image_64(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Impossible de lire l'image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img64 = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    img_up = cv2.resize(img64, (256, 256), interpolation=cv2.INTER_CUBIC)
    return img64, img_up

def variance_of_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def left_right_symmetry(face_rgb):
    # visage  (H, W, 3)
    h, w, _ = face_rgb.shape
    left = face_rgb[:, :w // 2]
    right = face_rgb[:, w - w // 2:]
    right_flipped = cv2.flip(right, 1)
    if left.shape[1] != right_flipped.shape[1]:
        minw = min(left.shape[1], right_flipped.shape[1])
        left = left[:, :minw]
        right_flipped = right_flipped[:, :minw]
    diff = (left.astype(np.float32) - right_flipped.astype(np.float32)) / 255.0
    mse = np.mean(diff ** 2)
    return max(0.0, 1.0 - (mse * 10.0))  # facteur empirique

def skin_mask_ycrcb(img_rgb):
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)
    mask = cv2.inRange(img_ycrcb, (0, 135, 85), (255, 180, 135))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

def artifacts_score(face_rgb):
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    hf = variance_of_laplacian(gray)
    p0 = np.mean(gray <= 2)
    p255 = np.mean(gray >= 253)
    extremes = p0 + p255
    hf_norm = np.tanh(hf / 300.0)  # 0..~1
    extreme_penalty = min(1.0, extremes * 10.0)
    score = hf_norm * (1.0 - 0.7 * extreme_penalty)
    return np.clip(score, 0.0, 1.0)

def eyes_quality(face_rgb, eyes_cascade):
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    roi = gray[0:int(0.6 * h), :]
    eyes = eyes_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
    if len(eyes) == 0:
        return 0.0
    # Heuristique : 2 yeux ~ meilleur, vérifie alignement horizontal
    best = 0.3  # au moins on a détecté quelque chose
    if len(eyes) >= 2:
        # prendre deux plus grands
        eyes_sorted = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
        (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes_sorted
        # alignement vertical: plus proche -> meilleur
        dy = abs((y1 + h1/2) - (y2 + h2/2)) / h
        # distance horizontale raisonnable
        dx = abs((x1 + w1/2) - (x2 + w2/2)) / w
        align_score = max(0.0, 1.0 - dy * 3.0)
        spacing_score = 1.0 if 0.15 <= dx <= 0.6 else max(0.0, 1.0 - abs(dx - 0.35) * 4.0)
        best = 0.6 * align_score + 0.4 * spacing_score
        best = np.clip(best, 0.0, 1.0)
    return float(best)

def skin_coverage_quality(face_rgb):
    mask = skin_mask_ycrcb(face_rgb)
    cov = np.mean(mask > 0)  # 0..1
    if cov < 0.2 or cov > 0.98:
        return 0.1
    # Score maximal si couverture entre 0.4 et 0.9
    if 0.4 <= cov <= 0.9:
        return 1.0
    if cov < 0.4:
        return 0.1 + 0.9 * (cov - 0.2) / (0.2)
    else:
        return 0.1 + 0.9 * (0.98 - cov) / (0.08)

def credibility_score(face_rgb, eyes_cascade):
    sym = left_right_symmetry(face_rgb)
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    sharp = np.tanh(variance_of_laplacian(gray) / 300.0)  # 0..1
    eyes = eyes_quality(face_rgb, eyes_cascade)
    skinq = skin_coverage_quality(face_rgb)
    arti = artifacts_score(face_rgb)

    w_sym, w_sharp, w_eyes, w_skin, w_arti = 0.25, 0.20, 0.25, 0.15, 0.15
    score01 = (w_sym*sym + w_sharp*sharp + w_eyes*eyes + w_skin*skinq + w_arti*arti)
    return float(np.clip(score01 * 100.0, 0.0, 100.0)), {
        "symmetry": round(sym, 3),
        "sharpness": round(sharp, 3),
        "eyes": round(eyes, 3),
        "skin": round(skinq, 3),
        "artifacts": round(arti, 3)
    }

def detect_face(img_up, face_cascade):
    gray = cv2.cvtColor(img_up, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
    )
    if len(faces) == 0:
        return None
    # prendre le plus grand rectangle
    x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
    # sécuriser un carré un peu plus large
    pad = int(0.10 * max(w, h))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(img_up.shape[1], x + w + pad)
    y1 = min(img_up.shape[0], y + h + pad)
    face_rgb = img_up[y0:y1, x0:x1]
    # Normaliser le visage à 160x160 pour les mesures
    face_rgb = cv2.resize(face_rgb, (160, 160), interpolation=cv2.INTER_CUBIC)
    return face_rgb, (int(x0), int(y0), int(x1 - x0), int(y1 - y0))

def get_score(path):

    img64, img_up = load_image_64(path)

    haar_dir = cv2.data.haarcascades
    face_xml = os.path.join(haar_dir, "haarcascade_frontalface_default.xml")
    eyes_xml = os.path.join(haar_dir, "haarcascade_eye.xml")
    face_cascade = cv2.CascadeClassifier(face_xml)
    eyes_cascade = cv2.CascadeClassifier(eyes_xml)

    if face_cascade.empty() or eyes_cascade.empty():
        raise RuntimeError("Impossible de charger les Haar cascades d'OpenCV.")

    detection = detect_face(img_up, face_cascade)
    result = {
        "path": path,
        "detected_face": False,
        "credibility_score": None,
        "subscores": None,
        "face_box_upscaled": None  # (x,y,w,h) sur l'image upscalée 256x256
    }

    if detection is None:
        result["detected_face"] = False
        #print(f"[ {path} ] Visage: NON | Score: N/A")
        return 0

    face_rgb, box = detection
    result["detected_face"] = True
    result["face_box_upscaled"] = box
    score, subs = credibility_score(face_rgb, eyes_cascade)
    result["credibility_score"] = round(score, 1)
    result["subscores"] = subs


    #print(f"[{path}] Visage: OUI | Score crédibilité: {result['credibility_score']}/100")
    #print(f"  Détails -> symmetry: {subs['symmetry']}, sharpness: {subs['sharpness']}, eyes: {subs['eyes']}, skin: {subs['skin']}, artifacts: {subs['artifacts']}")
    #print(f"  Face box (256x256): {box}")

    return 100+score


def get_dict_scores(base_path):
    l_models = os.listdir(base_path)
    result = {}

    for l_model in l_models:
        model_path = base_path + "/" + l_model
        epochs_list = [int(e) for e in os.listdir(model_path)]
        epochs_list.sort()
        print("epochs : ", epochs_list)

        result[l_model] = {}

        for current_epoch_i in epochs_list:
            current_epoch = str(current_epoch_i)

            l_images = os.listdir(model_path + "/" + current_epoch)

            scores_history = []

            print("==> current : ", model_path + "/" + current_epoch)


            for current_image in l_images:
                current_path = model_path + "/" + current_epoch + "/" + current_image

                score = get_score(current_path)

                # print("===> Current img",current_path, "obtained ",score)
                scores_history.append(score)

            scores_history.sort()

            print("scores : ", scores_history)

            #result[l_model + "/" + str(current_epoch_i)] = scores_history
            result[l_model][str(current_epoch_i)] = scores_history
    return result

def plot_percentage_fail(res):
    for model_name in res.keys():
        current_line=[]
        for epoch in res[model_name].keys():
            ctr = 0
            for image in res[model_name][epoch]:
                if image==0:
                    ctr+=1
            current_line.append(ctr/len(res[model_name][epoch]))
        plt.plot(current_line,label=model_name)
    plt.title("Evolution of failed images")
    plt.xlabel("Epochs")
    plt.ylabel("Fail proportion")
    plt.legend()
    plt.show()

def plot_score_success(res):
    for model_name in res.keys():
        current_line=[]
        for epoch in res[model_name].keys():
            total = 0
            for image in res[model_name][epoch]:
                if image!=0:
                    total+=(image-100)
            current_line.append(total/len(res[model_name][epoch]))
        plt.plot(current_line,label=model_name)
    plt.title("Evolution of score for images")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

base_path = "fake_images"

res = get_dict_scores(base_path)

print('====> Done 1')
for key in res.keys():
    #print('==> ',key,":",res[key])
    for key2 in res[key].keys():
        print('==> ',key2,":",key,":",res[key][key2])
print('====> Done2')

#plot_percentage_fail(res)
plot_score_success(res)