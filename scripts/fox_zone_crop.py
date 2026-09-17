"""Native-resolution RF-centered input for a separate FOX zone-only model."""
import math

SIZE=320


def crop_zone(image,prior):
    if image is None or image.shape[:2]!=(720,1280):
        raise ValueError('FOX crop refinement expects a native 1280×720 frame')
    def center(key,default,low,high):
        try:
            value=float(prior.get(key,default))
        except (TypeError,ValueError):
            return default
        return value if math.isfinite(value) and low<=value<=high else default
    x=center('zone_x',640,435,845);y=center('zone_y',300,170,425)
    left=int(max(0,min(1280-SIZE,round(x-SIZE/2))))
    top=int(max(0,min(720-SIZE,round(y-SIZE/2))))
    return image[top:top+SIZE,left:left+SIZE],dict(left=left,top=top,width=SIZE,height=SIZE)


def restore_zone(prediction,crop):
    """Convert a crop-coordinate detection back to the full broadcast frame."""
    try:
        x,y,w,h=[float(prediction[k]) for k in ('x','y','width','height')]
    except (KeyError,TypeError,ValueError):
        return None
    if not all(math.isfinite(v) for v in (x,y,w,h)) or w<=0 or h<=0:
        return None
    if x-w/2<0 or y-h/2<0 or x+w/2>crop['width'] or y+h/2>crop['height']:
        return None
    return dict(zone_x=x+crop['left'],zone_y=y+crop['top'],zone_width=w,zone_height=h)
