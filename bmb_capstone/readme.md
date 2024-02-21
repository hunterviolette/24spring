# Predictions page
Predict a segmented mask for a set of images
![PredPage](assets/predict_page.png)

# Upload page
Add image sets for training models, creating new masks, or out-of-sample testing 
![UploadPage](assets/upload_page.png)

## Enable Nvidia runtime inside docker containers 
- Open Docker -> settings -> docker engine 
```json
{
  "builder": {
    "gc": {
      "defaultKeepStorage": "20GB",
      "enabled": true
    }
  },
  "experimental": false,
  "features": {
    "buildkit": true
  },
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}