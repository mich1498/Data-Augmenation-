from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import random
import numpy as np

import cv2
import numbers
import collections

from utils import resize_image

# default list of interpolations
_DEFAULT_INTERPOLATIONS = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]



class Compose(object):
  """Composes several transforms together.

  Args:
      transforms (list of ``Transform`` objects): list of transforms to compose.

  Example:
      >>> Compose([
      >>>     Scale(320),
      >>>     RandomSizedCrop(224),
      >>> ])
  """
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, img):
    for t in self.transforms:
      img = t(img)
    return img

  def __repr__(self):
    repr_str = ""
    for t in self.transforms:
      repr_str += t.__repr__() + '\n'
    return repr_str

class RandomHorizontalFlip(object):
  """Horizontally flip the given numpy array randomly 
     (with a probability of 0.5).
  """
  def __call__(self, img):
    """
    Args:
        img (numpy array): Image to be flipped.

    Returns:
        numpy array: Randomly flipped image
    """
    if random.random() < 0.5:
      img = cv2.flip(img, 1)
      return img
    return img

  def __repr__(self):
    return "Random Horizontal Flip"

class Scale(object):
  """Rescale the input numpy array to the given size.

  Args:
      size (sequence or int): Desired output size. If size is a sequence like
          (w, h), output size will be matched to this. If size is an int,
          smaller edge of the image will be matched to this number.
          i.e, if height > width, then image will be rescaled to
          (size, size * height / width)

      interpolations (list of int, optional): Desired interpolation. 
      Default is ``CV2.INTER_NEAREST|CV2.INTER_LANCZOS|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
      Pass None during testing: always use CV2.INTER_LINEAR
  """
  def __init__(self, size, interpolations=_DEFAULT_INTERPOLATIONS):
    assert (isinstance(size, int) 
            or (isinstance(size, collections.Iterable) 
                and len(size) == 2)
           )
    self.size = size
    # use bilinear if interpolation is not specified
    if interpolations is None:
      interpolations = [cv2.INTER_LINEAR]
    assert isinstance(interpolations, collections.Iterable)
    self.interpolations = interpolations

  def __call__(self, img):
    """
    Args:
        img (numpy array): Image to be scaled.

    Returns:
        numpy array: Rescaled image
    """
    # sample interpolation method
    interpolation = random.sample(self.interpolations, 1)[0]
    h,w,c=img.shape
    #print('size of the image is %sx%s' % (w,h))
    #print(self.size)
    # scale the image
    if isinstance(self.size, int):
      
      if(h>w):
        img=cv2.resize(img,(self.size,math.floor(self.size*(h/w))),interpolation)
      else:
        img=cv2.resize(img,(math.floor(self.size*(w/h)),self.size),interpolation)
      
      return img
    else:
     
      w_new, h_new= self.size
      img=cv2.resize(img,(w_new,h_new),interpolation)
      
      return img

  def __repr__(self):
    if isinstance(self.size, int):
      target_size = (self.size, self.size)
    else:
      target_size = self.size
    return "Scale [Exact Size ({:d}, {:d})]".format(target_size[0], target_size[1])

class RandomSizedCrop(object):
  """Crop the given numpy array to random area and aspect ratio.

  A crop of random area of the original size and a random aspect ratio 
  of the original aspect ratio is made. This crop is finally resized to given size.
  This is widely used as data augmentation for training image classification models

  Args:
      size (sequence or int): size of target image. If size is a sequence like
          (w, h), output size will be matched to this. If size is an int, 
          output size will be (size, size).
      interpolations (list of int, optional): Desired interpolation. 
      Default is ``CV2.INTER_NEAREST|CV2.INTER_LANCZOS|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
      area_range (list of int): range of the areas to sample from
      ratio_range (list of int): range of aspect ratio to sample from
      num_trials (int): number of sampling trials
  """

  def __init__(self, size, interpolations=_DEFAULT_INTERPOLATIONS, 
               area_range=(0.25, 1.0), ratio_range=(0.8, 1.2), num_trials=10):
    self.size = size
    if interpolations is None:
      interpolations = [cv2.INTER_LINEAR]
    assert isinstance(interpolations, collections.Iterable)
    self.interpolations = interpolations
    self.num_trials = int(num_trials)
    self.area_range = area_range
    self.ratio_range = ratio_range

  def __call__(self, img):
    # sample interpolation method
    interpolation = random.sample(self.interpolations, 1)[0]
    for attempt in range(self.num_trials):


      area = img.shape[0] * img.shape[1]
      target_area = random.uniform(self.area_range[0], self.area_range[1])*area
      aspect_ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])

      
      h,w,c=img.shape
      #using the aspect ratio and the target area, we get 2 equations
      #assuming aspect is w/h
      h1=int(math.sqrt(target_area/aspect_ratio))
      w1= int(aspect_ratio*h1)

      #assuming the aspect ratio is h/w
      w2= int(math.sqrt(target_area/aspect_ratio))
      h2=int(w2*aspect_ratio)
      #in order for the patch to be acceptable both the height and width have to be
      if (w1<= w and h1 <=h):
        start_w=(w - w1)//2
        start_h=(h - h1)//2
        img = img[start_h:start_h + h1, start_w: start_w + w1]  ## cropped image
        img = img * 255
        img = img.astype(np.uint8)
        img=resize_image(img, (self.size, self.size), interpolation)
        return img
      if (w2<=w and h2<=h):
        start_w = (w - w2) // 2
        start_h = (h - h2) // 2
        img = img[start_h:start_h + h2, start_w: start_w + w2]  ## cropped image
        img = img * 255
        img = img.astype(np.uint8)
        img=resize_image(img, (self.size, self.size), interpolation)
        return img

    # Fall back
    if isinstance(self.size, int):
      im_scale = Scale(self.size, interpolations=self.interpolations)
      img = im_scale(img)
      start_w = (w - self.size) // 2
      start_h = (h - self.size) // 2
      img = img[start_h:start_h + self.size, start_w: start_w + self.size]  ## cropped image
      # with a square sized output, the default is to crop the patch in the center 
      # (after all trials fail)
      return img
    else:
      # with a pre-specified output size, the default crop is the image itself
      im_scale = Scale(self.size, interpolations=self.interpolations)
      img = im_scale(img)
      return img

  def __repr__(self):
    if isinstance(self.size, int):
      target_size = (self.size, self.size)
    else:
      target_size = self.size
    return "Random Crop" + \
           "[Size ({:d}, {:d}); Area {:.2f} - {:.2f}%; Ratio {:.2f} - {:.2f}%]".format(
            target_size[0], target_size[1], 
            self.area_range[0], self.area_range[1],
            self.ratio_range[0], self.ratio_range[1])


class RandomColor(object):
  """Perturb color channels of a given image
  Sample alpha in the range of (-r, r) and multiply 1 + alpha to a color channel. 
  The sampling is done independently for each channel.

  Args:
      color_range (float): range of color jitter ratio (-r ~ +r) max r = 1.0
  """
  def __init__(self, color_range):
    self.color_range = color_range


  def __call__(self, img):
    #################################################################################
    ## so from my understanding, color_range is a float with max value =1.0
    ##we randomly sample a value from this range
    ##after sampling , we multiply this to each color channel
    #################################################################################
    alpha = random.uniform(-self.color_range, self.color_range)

    ##split individual color channels and multiply with (1+alpha to each color channel)
    r = img[:, :, 0] * (1 + alpha)
    g = img[:, :, 1] * (1 + alpha)
    b = img[:, :, 2] * (1 + alpha)
    #merge all 3 color channels
    img = cv2.merge((r, g, b))
    #clipping values greater than 255 to 255
    imgflat=img.flatten()
    for i in range(len(imgflat)):
      if(imgflat[i]>=255):
        imgflat[i]=255
    img=np.reshape(imgflat,img.shape)
    img=img/255 #dividing by 255 so pixel values lie between 0 to 1
    return img

  def __repr__(self):
    return "Random Color [Range {:.2f} - {:.2f}%]".format(
            1-self.color_range, 1+self.color_range)


class RandomRotate(object):
  """Rotate the given numpy array (around the image center) by a random degree.

  Args:
      degree_range (float): range of degree (-d ~ +d)
  """
  def __init__(self, degree_range, interpolations=_DEFAULT_INTERPOLATIONS):
    self.degree_range = degree_range
    if interpolations is None:
      interpolations = [cv2.INTER_LINEAR]
    assert isinstance(interpolations, collections.Iterable)
    self.interpolations = interpolations

  def __call__(self, img):
    # sample interpolation method
    interpolation = random.sample(self.interpolations, 1)[0]
    # sample rotation
    degree = random.uniform(-self.degree_range, self.degree_range)
    # ignore small rotations
    if np.abs(degree) <= 1.0:
      return img


    image_size = (img.shape[1], img.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
      [cv2.getRotationMatrix2D(image_center, degree, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])


    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
      (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
      (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
      (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
      (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
      [1, 0, int(new_w * 0.5 - image_w2)],
      [0, 1, int(new_h * 0.5 - image_h2)],
      [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
      img,
      affine_mat,
      (new_w, new_h),
      flags=cv2.INTER_LINEAR
    )

    
    #to compute the width and height of the largest rotated rectangle in the image
    h,w,c=img.shape
    angle= math.radians(degree)
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3 ##
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)
    width= bb_w - 2 * x
    height=bb_h - 2 * y
    ###################################################################################
    # to crop the image
    image_size=(result.shape[1],result.shape[0])
    image_center=(int(image_size[0]*0.5),int(image_size[1]*0.5))
    if (width > image_size[0]):
        width = image_size[0]

    if (height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)
    img=result[y1:y2,x1:x2]
    return img

  def __repr__(self):
    return "Random Rotation [Range {:.2f} - {:.2f} Degree]".format(
            -self.degree_range, self.degree_range)