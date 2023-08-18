import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
import base64

app = FastAPI()

# Enable CORS for all domains
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def ghog(img):
    img_new = cv2.resize(img, (128, 128), cv2.INTER_AREA)
    win_size = (128, 128)
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    hog_descriptor = hog.compute(img_new)
    return hog_descriptor.tolist()

@app.get("/api/genhog")
async def read_str(img: str = Query(...)):
    # Decode base64 image data
    img_data = img.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    hog = ghog(img)
    return {"HOG Descriptor": hog}