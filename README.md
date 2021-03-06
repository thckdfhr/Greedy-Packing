# Greedy-Packing
### bin packing problem

```
This code is for bin-packing problem with recatnagle boxes in image.   
This code greedly pack all the boxes in small region.   
The key idea is packing two box, repeatedly by specific order.   
```   


#### First size difference makes overhead so divide groups with big size difference.   

![image](https://user-images.githubusercontent.com/48256991/147639069-4ca107c2-6639-42f2-aad0-bf0dbf9ad289.png)


#### Then shape of recatangles makes overhead so add the box by vertically or horizontally by the shape of box.   
![image](https://user-images.githubusercontent.com/48256991/147639246-4d7b0ae2-4edc-40ba-8a56-e054ff90c05c.png)

* * *
* * *
* * *


input: img([B,C,H,W]), 　box_to_pack([N,4],　 None, 　None,　 None   
output : packed_img([B,C,H,W]),　 where_old_image, 　where_new_image,　 groups_groups
   
   
```
img : original image 　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　// Tensor [B,C,H,W]   
box_to_pack : The recatangle boxes to pack from original image　　　　　　　　　　　　　　　　　　　　　　　　　　　 // Tensor (N box with [x_min,y_min,x_max,y_max])   
packed_img : Greedy packed imaged comprising box_to_pack 　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　// Tensor [B,C,H,W]   
where_old_image : The cutted place of original image for greedy pack 　　　　　　　　　　　　　　　　　　　　　　　　// Tensor [M,4]   
where_new_image : The laid place on packed image by greedy pack (orders match with where_old_image)　　　　　　　　 // Tensor [M,4]   
groups_groups : The indexes of boxes which made the cutted place　　　　　　　　　　　　　　　　　　　　　　　　　　 // List len M   
```   
   
   
* * *
* * *

code implement: 
```
packed_img, where_old_image, where_new_image, groups_groups = pack2(img, box_to_pack, None, None, None)
```


![image](https://user-images.githubusercontent.com/48256991/147637738-21b6e1e9-ade5-4e43-9bf0-45a5b7bcbcb2.png)

* * *
* * *
example:

![image](https://user-images.githubusercontent.com/48256991/147637932-06344195-ec92-400a-ae96-214e8a8c5ce7.png)   
![image](https://user-images.githubusercontent.com/48256991/147637938-570cd82a-b212-4600-a459-5c88816775e8.png)
