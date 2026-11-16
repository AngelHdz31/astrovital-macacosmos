import os
import urllib.request
from PIL import Image

# 1. Crear la carpeta si no existe
os.makedirs("data/muestras_vision", exist_ok=True)

# 2. URLs de imágenes médicas y faciales reales (Wikimedia Commons)
urls = {
    "huesos_fractura_demo.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/X-ray_of_normal_forearm.jpg/640px-X-ray_of_normal_forearm.jpg",
    "tomografia_cerebral_demo.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/Ct-scan_of_the_brain.jpg/640px-Ct-scan_of_the_brain.jpg",
    "animo_fatiga_demo.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Laughter.jpg/640px-Laughter.jpg"
}

# 3. Descargar con un User-Agent de navegador para evitar el Error 403
print("Descargando muestras de visión computacional...")
for nombre_archivo, url in urls.items():
    ruta_final = os.path.join("data/muestras_vision", nombre_archivo)
    print(f" -> Procesando {nombre_archivo}")
    
    # Crear la petición simulando ser Google Chrome
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
    
    with urllib.request.urlopen(req) as response, open("temp_img.jpg", 'wb') as out_file:
        out_file.write(response.read())
    
    # Asegurar compatibilidad estricta con PNG
    img = Image.open("temp_img.jpg").convert("RGB")
    img.save(ruta_final, "PNG")

if os.path.exists("temp_img.jpg"):
    os.remove("temp_img.jpg")

print("✅ ¡Muestras instaladas! Ya puedes ir a la pestaña '5. Visión Computacional (IA)' y probarlas.")
