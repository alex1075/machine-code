# Machine-code repository for Dx device project
Ignore below, to be updated

### How to prepare the data for ML analysis
# 1. Process the data
To breakdown the videos into idividual frames, we use the following command from the root folder: 
``` bash
python -c "from code.convert import convertVideoToImages; convertVideoToImages('Data/Videos/', 'Data/Image_temp/')"
```
Then resize the frames to a fixed size.
``` bash
python -c "from code.convert import resizeAllJpg; resizeAllJpg('Data/Image_temp/', (1080, 1920))"
```
Then random crop the images. 
``` bash
python -c "from code.convert import radomCrop; randomCrop('Data/Image_temp/', 'Data/Images/', (256,256))"
```
Now label the images cropped. 
# 2. Split the data into training and test sets
# 3. Normalize the data

``` bash
python -c "from code.convert import *; iterateBlur()"
```
``` bash
python -c "from code.convert import *; detectBlurr()"

```
``` bash
python -c "from code.convert import *; chopUpDataset()"
```