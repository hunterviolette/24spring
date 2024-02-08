# Predictions page
![PredPage](assets/predPage.png)

# Assumptions
- Input images or masks are .png or .tif
- Assumes data loaded has image/mask pair
  1. if img is named x, mask named x_mask
  2. Finds mask file by: x.replace(".", "_mask.")