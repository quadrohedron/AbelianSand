import math
import numpy as np
from PIL import Image, ImageDraw, ImagePalette

class ImagerSingleton:
    palettes = dict()
    
    meta_colours = [(i, i, i) for i in (128, 254, 1)]+[(100, i, 100) for i in range(4)]+[(128, 0, 0)]
    rainbow_value = 0.85
    rainbow_hue_shift = 0.4
    zero_value = 100
    
    def __init__(self, palette_fp = None):
        P = ImagePalette.ImagePalette()
        for i in range(256):
            P.getcolor(tuple(i for _ in range(3)))
        self.palettes['grayscale'] = P
        clrs = [(80, 80, 80), (177, 0, 255), (44, 0, 255),
                 (0, 222, 255), (0, 255, 22), (244, 255, 0),
                 (255, 133, 0), (255, 0, 0)]
        P = ImagePalette.ImagePalette()
        for c in self.meta_colours:
            P.getcolor(c)
        for c in clrs:
            P.getcolor(c)
        for i in range(1,241):
            v = min(8*i, 255)
            P.getcolor((255, v, v))
        self.palettes['default'] = P
        #
        for i in (3, 4, 6, 8, 12):
            self.palettes['r'+str(i)] = self.make_rainbow(i)
        #
        # Implement palette importing
    
    def hsv2rgb(h, s, v): # ~1 -> ~255
        hi = math.floor(6*h)%6
        vmin = (1-s)*v
        a = (v-vmin)*((6*h)%1)
        if hi == 0:
            res = (v, vmin+a, vmin)
        elif hi == 1:
            res = (v-a, v, vmin)
        elif hi == 2:
            res = (vmin, v, vmin+a)
        elif hi == 3:
            res = (vmin, v-a, v)
        elif hi == 4:
            res = (vmin+a, vmin, v)
        elif hi == 5:
            res = (v, vmin, v-a)
        return tuple(round(255*x) for x in res)
    
    def make_rainbow(self, sector_num):
        v, hs = self.rainbow_value, self.rainbow_hue_shift
        p = ImagePalette.ImagePalette()
        for c in self.meta_colours:
            p.getcolor(c)
        p.getcolor(tuple(self.zero_value for _ in range(3)))
        hvals = tuple((hs-(1+i)/sector_num)%1.0 for i in range(sector_num))
        for i in range(255-len(p.colors)):
            p.getcolor(ImagerClass.hsv2rgb(hvals[i%sector_num], pow(0.5, i//sector_num), v))
        return p
    
    def paint_asp(self, asp, palette = None, cell_size = None, image_size = None):
        if cell_size == image_size == None:
            raise ValueError('Neither cell size nor image size hint specified.')
        if palette == None:
            palette = 'r'+str(asp.limit)
        if not (palette in self.palettes):
            if palette[0] == 'r' and palette[1:].isdigit():
                self.palettes[palette] = self.make_rainbow(int(palette[1:]))
            else:
                raise KeyError(f'Palette \'{palette}\' not found.')
                return None
        reqattr = 'paint'+str(asp.cell_shape)
        if hasattr(self, reqattr):
            return getattr(self, reqattr)(asp, palette, cell_size = cell_size, image_size = image_size)
        else:
            raise NotImplementedError(f'No paint method for cell shape {asp.cell_shape}.')
    
    def paint3(self, asp, palette, cell_size = None, image_size = None):
        p = self.palettes[palette].copy()
        c = asp.get_plane()
        ph, pw = c.shape
        if image_size:
            cell_size = 2*round(image_size/(pw+1))
        uh, huw = round(cell_size*math.sqrt(3)/2), cell_size//2
        ih, iw = uh*ph, 2*huw*ph
        im = Image.new('P', (iw, ih))
        im.palette = p
        maxv = len(im.palette.colors)-1
        dr = ImageDraw.Draw(im)
        oy, ox = 0, iw//2
        for i in range(ph):
            for j in range(ph):
                v = int(min(8+c[i, 2*j], maxv))
                px, py = ox+huw*(j-i), oy+uh*(j+i)
                dr.polygon([(px, py), (px+huw, py+uh), (px-huw, py+uh)], v)
            for j in range(ph-1):
                v = int(min(8+c[i, 2*j+1], maxv))
                px, py = ox+huw*(j-i), oy+uh*(2+j+i)
                dr.polygon([(px, py), (px-huw, py-uh), (px+huw, py-uh)], v)
        return im
    
    def paint4(self, asp, palette, cell_size = None, image_size = None):
        p = self.palettes[palette].copy()
        maxv = len(p.colors)-1
        c = asp.get_plane()
        c = np.vectorize(lambda x: x+8 if x+8 < maxv else maxv)(c)
        im = Image.fromarray(c.astype('uint8'), mode = 'P')
        im.palette = p
        m, n = c.shape
        if image_size:
            cell_size = round(image_size/max(m, n))
        return im.resize((cell_size*n, cell_size*m))
    
    def paint6(self, asp, palette, cell_size = None, image_size = None):
        p = self.palettes[palette].copy()
        c = asp.get_plane()
        d = c.shape[0]
        sl = (d+1)//2
        if image_size:
            cell_size = 4*round(image_size/(2*d*math.sqrt(3)))
        iu, ru = round(cell_size*math.sqrt(3)/4), round(cell_size/4)
        iw, ih = 2*d*iu+1, 2*(d+sl)*ru+1
        im = Image.new('P', (iw, ih))
        im.palette = p
        maxv = len(im.palette.colors)-1
        dr = ImageDraw.Draw(im)
        ox, oy = iu, 3*(sl-1)*ru
        shape = ((0, 0), (iu, ru), (iu, 3*ru),
                 (0, 4*ru), (-iu, 3*ru), (-iu, ru))
        for i in range(d):
            a = 1-sl+i
            jrange = range(max(0,a),min(d,d+a))
            for j in jrange:
                v = int(min(8+c[i, j], maxv))
                px, py = ox+(i+j)*iu, oy+(j-i)*3*ru
                dr.polygon([(px+vx, py+vy) for vx, vy in shape], v)
        return im
