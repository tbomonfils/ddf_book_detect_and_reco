# Repo aiming at detecting and recognizing books from their cover

## Technical corrections left

- [ ] Check images modifications that couldn't be done (logs)
- [ ] Too many backforth between image.array and PIL.image
- [ ] Mix of data augmentation tools, is is possible to select one ?
- [ ] The transformation matrix include a translation after rotation on x or y axe, the back translation from the homogenous coordinate system (projective coordinates) need a correction ?