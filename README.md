# Repo aiming at detecting and recognizing books from their cover

## Technical corrections left

- [x] Check images modifications that couldn't be done (logs)
- [ ] Cover img out of background img ?
- [ ] Too many back and forth between image.array and PIL.image
- [ ] Mix of data augmentation tools, is is possible to select one ?
- [ ] The transformation matrix include a translation after rotation on x or y axe, the back translation from the homogenous coordinate system (projective coordinates) need a correction ?
- [ ] Over head of multi scale and yolo's head
- [ ] Data augmentation : more variation of backgrounds using Coco ? Some are too Dark ?
- [ ] There is a black thin line around the incrusted cover, can it be removed ?
- [ ] Use real img for training/eval ?