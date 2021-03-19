import numpy as np
import time

from .SandImager import ImagerSingleton

class BaseASP:
    active:bool
    cell_shape:int
    content:np.ndarray
    evo_time = 0.0
    extras = dict()
    fold = 1
    limit:int
    mask_master:np.ndarray
    perf = time.perf_counter
    steps = 0
    
    def __init__(self, content, **params):
        if not type(content) == np.ndarray or not 'int' in content.dtype.name:
            content = np.array(content, dtype = int)
        if not (content == abs(content)).all():
            self.mask_master = ((content//abs(content))+1)//2
            self.content = content*self.mask_master
        else:
            self.mask_master = np.ones_like(content, dtype = int)
            self.content = content.copy()
        for k in params:
            if hasattr(self, k):
                setattr(self, k, params[k])
    
    def __eq__(self, obj):
        if self.iscongruent(obj):
            return (self.get_plane() == obj.get_plane()).all()
        return False
    
    def copy(self, t = None, **params):
        if t:
            try:
                return t(self.content, **params)
            except:
                pass
        return type(self)(self.content, **params)
    
    def format_time(self, t):
        t = round(t)
        res = []
        for d in (60, 60, 24, 1):
            t, v = divmod(t, d)
            res.insert(0, ('0' if v < 10 else '')+str(v))
        res[0] = res[0].lstrip('0')
        if res[0] == '':
            res[0] = '0'
        return ':'.join(res)
    
    def evo_step(self):
        pass
    
    def evolve(self, ret = False):
        while self.active:
            self.evo_step()
        if ret:
            return self
    
    def get_plane(self):
        return self.content
    
    def iscongruent(self, obj):
        ts, to = type(self), type(obj)
        if ts == to:
            return self.content.shape == obj.content.shape
        if issubclass(ts, BaseASP) and issubclass(to, BaseASP) and self.cell_shape == obj.cell_shape:
            return self.get_plane().shape == obj.get_plane().shape
        return False
    
    def parse_plane(self, plane):
        return plane

class ExpandableASP(BaseASP):
    exp_time = 0.0

    def __init__(self, content, **params):
        super().__init__(content, **params)
        self.extras['expandable'] = True
        self.expand()
    
    def evolve(self, ret = False):
        while self.active:
            self.tot_step()
        if ret:
            return self

    def expand(self):
        pass

    def tot_step(self):
        self.evo_step()
        self.expand()

class StackASP(ExpandableASP):
    def __init__(self, height, **params):
        super().__init__(height*np.ones((1, 1), dtype = int), **params)
        self.expand()
        self.extras['height'] = height
    
    def expand(self):
        t0 = self.perf()
        h, w = self.content.shape
        if self.content[-1, 0].any():
            self.content = np.hstack((np.vstack((self.content, np.zeros((1, w), dtype = int))),
            np.zeros((h+1, 1), dtype = int)))
            h += 1
        self.exp_time += self.perf()-t0



def flat_asp(asp_type, value, shape):
    if type(shape) == int:
        cs = asp_type.cell_shape
        if cs == 3:
            shape = (shape, 2*shape-1)
        elif cs == 4:
            shape = (shape, shape)
        elif cs == 6:
            shape = (2*shape-1, 2*shape-1)
    return asp_type(value*np.ones(shape))

def max_asp(asp_type, shape):
    return flat_asp(asp_type, asp_type.limit, shape)

def unit_asp(asp_type, shape):
    return flat_asp(asp_type, 1, shape)
    


class N4W4F1(BaseASP):
    cell_shape = 4
    limit = 4
    masked:bool

    def __init__(self, content, **params):
        super().__init__(content, **params)
        self.active = (self.content//self.limit).any()
        self.masked = 0 in self.mask_master
    
    def evo_step(self):
        t0 = self.perf()
        d = self.content//self.limit
        if d.any():
            self.content -= self.limit*d
            self.content[:-1, :] += d[1:, :]
            self.content[1:, :] += d[:-1, :]
            self.content[:, :-1] += d[:, 1:]
            self.content[:, 1:] += d[:, :-1]
            if self.masked:
                self.content *= self.mask_master
            self.evo_time += self.perf()-t0
            self.steps += 1
        else:
            self.evo_time += self.perf()-t0
            self.active = False

class X4W4F1(N4W4F1, ExpandableASP):
    def __init__(self, content, **params):
        super().__init__(content, **params)
    
    def expand(self):
        t0 = self.perf()
        h, w = self.content.shape
        if self.content[0, :].any():
            self.content = np.vstack((np.zeros((1, w), dtype = int), self.content))
            h += 1
        if self.content[-1, :].any():
            self.content = np.vstack((self.content, np.zeros((1, w), dtype = int)))
            h += 1
        if self.content[:, 0].any():
            self.content = np.hstack((np.zeros((h, 1), dtype = int), self.content))
            w += 1
        if self.content[:, -1].any():
            self.content = np.hstack((self.content, np.zeros((h, 1), dtype = int)))
            w += 1
        self.exp_time += self.perf()-t0

class N4W4F4(N4W4F1):
    fold = 4

    def __init__(self, content, **params):
        super().__init__(content, **params)
    
    def evo_step(self):
        t0 = self.perf()
        d = self.content//self.limit
        if d.any():
            self.content -= self.limit*d
            self.content[:-1, :] += d[1:, :]
            self.content[1:, :] += d[:-1, :]
            self.content[:, :-1] += d[:, 1:]
            self.content[:, 1:] += d[:, :-1]
            self.content[:, 0] += d[:, 1]
            self.content[0, :] += d[1, :]
            if self.masked:
                self.content *= self.mask_master
            self.evo_time += self.perf()-t0
            self.steps += 1
        else:
            self.evo_time += self.perf()-t0
            self.active = False

    def get_plane(self):
        m, n = self.content.shape
        m -= 1
        n -= 1
        res = np.empty((2*m+1, 2*n+1), dtype = int)
        res[m:, n:] = self.content
        res[m:, :n] = self.content[:, :0:-1]
        res[:m, n:] = self.content[:0:-1, :]
        res[:m, :n] = self.content[:0:-1, :0:-1]
        return res
    
    def parse_plane(self, plane):
        h, w = plane.shape
        return plane[h//2:, w//2:]

class X4W4F4(N4W4F4, ExpandableASP):
    def __init__(self, content, **params):
        super().__init__(content, **params)
    
    def expand(self):
        t0 = self.perf()
        h, w = self.content.shape
        if self.content[-1, :].any():
            self.content = np.vstack((self.content, np.zeros((1, w), dtype = int)))
            h += 1
        if self.content[:, -1].any():
            self.content = np.hstack((self.content, np.zeros((h, 1), dtype = int)))
            w += 1
        self.exp_time += self.perf()-t0

class S4W4(StackASP, X4W4F4):
    def __init__(self, height, **params):
        super().__init__(height, **params)



class N4W8F1(BaseASP):
    cell_shape = 4
    limit = 8
    masked:bool

    def __init__(self, content, **params):
        super().__init__(content, **params)
        self.active = (self.content//self.limit).any()
        self.masked = 0 in self.mask_master
    
    def evo_step(self):
        t0 = self.perf()
        d = self.content//self.limit
        if d.any():
            self.content -= self.limit*d
            self.content[:-1, :] += d[1:, :]
            self.content[1:, :] += d[:-1, :]
            self.content[:, :-1] += d[:, 1:]
            self.content[:, 1:] += d[:, :-1]
            self.content[:-1, :-1] += d[1:, 1:]
            self.content[1:, :-1] += d[:-1, 1:]
            self.content[:-1, 1:] += d[1:, :-1]
            self.content[1:, 1:] += d[:-1, :-1]
            if self.masked:
                self.content *= self.mask_master
            self.evo_time += self.perf()-t0
            self.steps += 1
        else:
            self.evo_time += self.perf()-t0
            self.active = False

class X4W8F1(N4W8F1, X4W4F1):
    def __init__(self, content, **params):
        super().__init__(content, **params)

class N4W8F4(N4W8F1, N4W4F4):
    fold = 4

    def __init__(self, content, **params):
        super().__init__(content, **params)
    
    def evo_step(self):
        t0 = self.perf()
        d = self.content//self.limit
        if d.any():
            self.content -= self.limit*d
            self.content[:-1, :] += d[1:, :]
            self.content[1:, :] += d[:-1, :]
            self.content[:, :-1] += d[:, 1:]
            self.content[:, 1:] += d[:, :-1]
            self.content[:-1, :-1] += d[1:, 1:]
            self.content[1:, :-1] += d[:-1, 1:]
            self.content[:-1, 1:] += d[1:, :-1]
            self.content[1:, 1:] += d[:-1, :-1]
            #
            self.content[:, 0] += d[:, 1] #vw
            self.content[:-1, 0] += d[1:, 1] #vnw
            self.content[1:, 0] += d[:-1, 1] #vsw
            self.content[0, :] += d[1, :] #hn
            self.content[0, :-1] += d[1, 1:] #hnw
            self.content[0, 1:] += d[1, :-1] #hne
            self.content[0, 0] += d[1, 1] #qc
            #
            if self.masked:
                self.content *= self.mask_master
            self.evo_time += self.perf()-t0
            self.steps += 1
        else:
            self.evo_time += self.perf()-t0
            self.active = False

class X4W8F4(N4W8F4, X4W4F4):
    def __init__(self, content, **params):
        super().__init__(content, **params)

class S4W8(StackASP, X4W8F4):
    def __init__(self, height, **params):
        super().__init__(height, **params)



class N3W3F1(BaseASP):
    cell_shape = 3
    limit = 3

    mask_even:np.ndarray
    mask_odd:np.ndarray

    def __init__(self, content, **params):
        h, w = content.shape
        if not (2*h == w+1):
            raise ValueError('Content matrix is not pseudotriangular.')
            return None
        m = np.ones((h, w), dtype = int)
        for i in range(1, h):
            m[i, -2*i:] = 0
        super().__init__(content*m, **params)
        self.active = (self.content//self.limit).any()
        #
        self.mask_master *= m
        self.mask_even = np.ones((h, w), dtype = int)
        self.mask_even[:, 1::2] = 0
        self.mask_odd = 1-self.mask_even
        self.mask_even *= m
        self.mask_odd *= m
    
    def evo_step(self):
        t0 = self.perf()
        d = self.content//self.limit
        if d.any():
            self.content -= self.limit*d
            self.content[:, :-1] += d[:, 1:]
            self.content[:, 1:] += d[:, :-1]
            self.content[1:, :-1] += (d*self.mask_odd)[:-1, 1:]
            self.content[:-1, 1:] += (d*self.mask_even)[1:, :-1]
            self.content *= self.mask_master
            self.evo_time += self.perf()-t0
            self.steps += 1
        else:
            self.evo_time += self.perf()-t0
            self.active = False

class X3W3F1(N3W3F1, ExpandableASP):
    def __init__(self, content, **params):
        h, w = content.shape
        if not (2*h == w+1):
            raise ValueError('Content matrix is not pseudotriangular.')
            return None
        m = np.ones((h, w), dtype = int)
        for i in range(1, h):
            m[i, -2*i:] = 0
        self.mask_even = np.ones((h, w), dtype = int)
        self.mask_even[:, 1::2] = 0
        self.mask_odd = 1-self.mask_even
        self.mask_even *= m
        self.mask_odd *= m
        super().__init__(content, **params)
        h, w = self.mask_master.shape
        m = np.ones((h, w), dtype = int)
        for i in range(1, h):
            m[i, -2*i:] = 0
        self.mask_even = np.ones((h, w), dtype = int)
        #print(repr(self.mask_even))
        self.mask_even[:, 1::2] = 0
        self.mask_odd = 1-self.mask_even
        self.mask_even *= m
        self.mask_odd *= m
        #for m in (self.mask_master, self.mask_even, self.mask_odd):
                #print(repr(m))
    
    def expand(self):
        t0 = self.perf()
        h, w = self.content.shape
        unit = np.ones((1, 1), dtype = int)
        if self.content[0, ::2].any():
            t = np.zeros((1, w+2), dtype = int)
            r = np.zeros((h, 2), dtype = int)
            self.content = np.vstack((t, np.hstack((self.content, r))))
            self.mask_master = np.vstack((1-t, np.hstack((self.mask_master, r))))
            t[:, 1::2] = 1
            #print('t', repr(t))
            self.mask_odd = np.vstack((t, np.hstack((self.mask_odd, r))))
            self.mask_even = np.vstack((1-t, np.hstack((self.mask_even, r))))
            h += 1
            w += 2
            #for m in (self.mask_master, self.mask_even, self.mask_odd):
                #print(repr(m))
        if self.content[:, 0].any():
            l = np.zeros((h, 2), dtype = int)
            b = np.zeros((1, w+2), dtype = int)
            self.content = np.vstack((np.hstack((l, self.content)), b))
            mm_top = np.hstack((1-l, self.mask_master))
            mm_bot = np.hstack((unit, b[:, 1:]))
            self.mask_master = np.vstack((mm_top, mm_bot))
            l[:, 1] = 1
            mo_top = np.hstack((l, self.mask_odd))
            self.mask_odd = np.vstack((mo_top, b))
            me_top = np.hstack((1-l, self.mask_even))
            me_bot = np.hstack((unit, b[:, 1:]))
            self.mask_even = np.vstack((me_top, me_bot))
            h += 1
            w += 2
            #for m in (self.mask_master, self.mask_even, self.mask_odd):
                #print(repr(m))
        if self.content[:, ::-2].diagonal().any():
            r = np.zeros((h, 2), dtype = int)
            b = np.zeros((1, w+2), dtype = int)
            self.content = np.vstack((np.hstack((self.content, r)), b))
            mm_top = np.hstack((1-r, self.mask_master))
            mm_bot = np.hstack((unit, b[:, 1:]))
            self.mask_master = np.vstack((mm_top, mm_bot))
            r[:, 1] = 1
            mo_top = np.hstack((r, self.mask_odd))
            self.mask_odd = np.vstack((mo_top, b))
            me_top = np.hstack((1-r, self.mask_even))
            me_bot = np.hstack((unit, b[:, 1:]))
            self.mask_even = np.vstack((me_top, me_bot))
            h += 1
            w += 2
            #for m in (self.mask_master, self.mask_even, self.mask_odd):
                #print(repr(m))
        self.exp_time += self.perf()-t0

### TODO: Implement folded 3-way piles

class S3W3(StackASP, X3W3F1):
    def __init__(self, height, **params):
        super().__init__(height, **params)
    
    def expand(self): ### Here while folded expansion is not implemented
        X3W3F1.expand(self)



class N3W12F1(N3W3F1):
    limit = 12

    def __init__(self, content, **params):
        super().__init__(content, **params)
        self.active = (self.content//self.limit).any()
        self.masked = 0 in self.mask_master
    
    def evo_step(self):
        t0 = self.perf()
        d = self.content//self.limit
        if d.any():
            self.content -= self.limit*d
            self.content[1:, :] += d[:-1, :]
            self.content[1:, :-1] += d[:-1, 1:]
            self.content[1:, :-2] += d[:-1, 2:]
            self.content[:, :-1] += d[:, 1:]
            self.content[:, :-2] += d[:, 2:]
            self.content[:-1, :] += d[1:, :]
            self.content[:-1, 1:] += d[1:, :-1]
            self.content[:-1, 2:] += d[1:, :-2]
            self.content[:, 1:] += d[:, :-1]
            self.content[:, 2:] += d[:, :-2]
            e = d*self.mask_even
            self.content[:-1, :-1] += e[1:, 1:]
            self.content[:-1, 3:] += e[1:, :-3]
            o = d*self.mask_odd
            self.content[1:, 1:] += o[:-1, :-1]
            self.content[1:, :-3] += o[:-1, 3:]
            self.content *= self.mask_master
            self.evo_time += self.perf()-t0
            self.steps += 1
        else:
            self.evo_time += self.perf()-t0
            self.active = False

class X3W12F1(N3W12F1, X3W3F1):
    def __init__(self, content, **params):
        super().__init__(content, **params)
    
    def expand(self):
        t0 = self.perf()
        h, w = self.content.shape
        unit = np.ones((1, 1), dtype = int)
        if self.content[0, :].any():
            t = np.zeros((1, w+2), dtype = int)
            r = np.zeros((h, 2), dtype = int)
            self.content = np.vstack((t, np.hstack((self.content, r))))
            self.mask_master = np.vstack((1-t, np.hstack((self.mask_master, r))))
            t[:, 1::2] = 1
            #print('t', repr(t))
            self.mask_odd = np.vstack((t, np.hstack((self.mask_odd, r))))
            self.mask_even = np.vstack((1-t, np.hstack((self.mask_even, r))))
            h += 1
            w += 2
            #for m in (self.mask_master, self.mask_even, self.mask_odd):
                #print(repr(m))
        if self.content[:, :2].any():
            l = np.zeros((h, 2), dtype = int)
            b = np.zeros((1, w+2), dtype = int)
            self.content = np.vstack((np.hstack((l, self.content)), b))
            mm_top = np.hstack((1-l, self.mask_master))
            mm_bot = np.hstack((unit, b[:, 1:]))
            self.mask_master = np.vstack((mm_top, mm_bot))
            l[:, 1] = 1
            mo_top = np.hstack((l, self.mask_odd))
            self.mask_odd = np.vstack((mo_top, b))
            me_top = np.hstack((1-l, self.mask_even))
            me_bot = np.hstack((unit, b[:, 1:]))
            self.mask_even = np.vstack((me_top, me_bot))
            h += 1
            w += 2
            #for m in (self.mask_master, self.mask_even, self.mask_odd):
                #print(repr(m))
        if self.content[:, ::-2].diagonal().any():
            r = np.zeros((h, 2), dtype = int)
            b = np.zeros((1, w+2), dtype = int)
            self.content = np.vstack((np.hstack((self.content, r)), b))
            mm_top = np.hstack((1-r, self.mask_master))
            mm_bot = np.hstack((unit, b[:, 1:]))
            self.mask_master = np.vstack((mm_top, mm_bot))
            r[:, 1] = 1
            mo_top = np.hstack((r, self.mask_odd))
            self.mask_odd = np.vstack((mo_top, b))
            me_top = np.hstack((1-r, self.mask_even))
            me_bot = np.hstack((unit, b[:, 1:]))
            self.mask_even = np.vstack((me_top, me_bot))
            h += 1
            w += 2
            #for m in (self.mask_master, self.mask_even, self.mask_odd):
                #print(repr(m))
        self.exp_time += self.perf()-t0

class S3W12(StackASP, X3W12F1):
    def __init__(self, height, **params):
        super().__init__(height, **params)
    
    def expand(self): ### Here while folded expansion is not implemented
        X3W12F1.expand(self)



class N6W6F1(BaseASP):
    cell_shape = 6
    limit = 6

    def __init__(self, content, **params):
        if not type(content) == np.ndarray or not 'int' in content.dtype.name:
            content = np.array(content, dtype = int)
        h, w = content.shape
        if not h == w:
            raise ValueError('Content matrix is not shaped as a pseudocomb.')
            return None
        sl = (h+1)//2
        m = np.ones((h, h), dtype = int)
        for i in range(sl-1):
            m[i, 1-sl+i:] = 0
            m[-1-i, :sl-1-i] = 0
        super().__init__(content*m, **params)
        self.active = (self.content//self.limit).any()
        #
        self.mask_master *= m
    
    def evo_step(self):
        t0 = self.perf()
        d = self.content//self.limit
        if d.any():
            self.content -= self.limit*d
            self.content[:, :-1] += d[:, 1:]
            self.content[:, 1:] += d[:, :-1]
            self.content[:-1, :] += d[1:, :]
            self.content[1:, :] += d[:-1, :]
            self.content[:-1, :-1] += d[1:, 1:]
            self.content[1:, 1:] += d[:-1, :-1]
            self.content *= self.mask_master
            self.evo_time += self.perf()-t0
            self.steps += 1
        else:
            self.evo_time += self.perf()-t0
            self.active = False

class X6W6F1(N6W6F1, ExpandableASP):
    def __init__(self, content, **params):
        super().__init__(content, **params)
    
    def expand(self): ### Symmetrical only for now
        t0 = self.perf()
        d = self.content.shape[0]+2
        if self.content[0, :].any():
            c = np.zeros((d, d), dtype = int)
            c[1:-1, 1:-1] = self.content
            self.content = c
            self.mask_master = np.ones((d, d), dtype = int)
            sl = (d+1)//2
            for i in range(sl-1):
                self.mask_master[i, 1-sl+i:] = 0
                self.mask_master[-1-i, :sl-1-i] = 0
        self.exp_time += self.perf()-t0
        '''t0 = self.perf()
        h, w = self.content.shape
        unit = np.ones((1, 1), dtype = int)
        if self.content[0, :].any():
            t = np.zeros((1, w+2), dtype = int)
            r = np.zeros((h, 2), dtype = int)
            self.content = np.vstack((t, np.hstack((self.content, r))))
            self.mask_master = np.vstack((1-t, np.hstack((self.mask_master, r))))
            t[::2] = 1
            self.mask_odd = np.vstack((t, np.hstack((self.mask_odd, r))))
            self.mask_even = np.vstack((1-t, np.hstack((self.mask_even, r))))
            h += 1
            w += 2
        if self.content[:, 0].any():
            l = np.zeros((h, 2), dtype = int)
            b = np.zeros((1, w+2), dtype = int)
            self.content = np.vstack((np.hstack((l, self.content)), b))
            mm_top = np.hstack((1-l, self.mask_master))
            mm_bot = np.hstack((unit, b[:, 1:]))
            self.mask_master = np.vstack((mm_top, mm_bot))
            l[:, 1] = 1
            mo_top = np.hstack((l, self.mask_odd))
            self.mask_odd = np.vstack((mo_top, b))
            me_top = np.hstack((1-l, self.mask_even))
            me_bot = np.hstack((unit, b[:, 1:]))
            self.mask_even = np.vstack((me_top, me_bot))
            h += 1
            w += 2
        if self.content[:, ::-2].diagonal().any():
            r = np.zeros((h, 2), dtype = int)
            b = np.zeros((1, w+2), dtype = int)
            self.content = np.vstack((np.hstack((self.content, r)), b))
            mm_top = np.hstack((1-r, self.mask_master))
            mm_bot = np.hstack((unit, b[:, 1:]))
            self.mask_master = np.vstack((mm_top, mm_bot))
            r[:, 1] = 1
            mo_top = np.hstack((r, self.mask_odd))
            self.mask_odd = np.vstack((mo_top, b))
            me_top = np.hstack((1-r, self.mask_even))
            me_bot = np.hstack((unit, b[:, 1:]))
            h += 1
            w += 2
        self.exp_time += self.perf()-t0'''

class N6W6F12(N6W6F1):
    fold = 12

    def __init__(self, content, **params):
        if not type(content) == np.ndarray or not 'int' in content.dtype.name:
            content = np.array(content, dtype = int)
        h, w = content.shape
        if not (w-1 <= 2*h <= w):
            if not (h == 1 and w in (1, 2)):   
                raise ValueError('Content matrix is not shaped as a 12-fold pseudocomb.')
                return None
        m = np.ones((h, w), dtype = int)
        for i in range(1, h):
            m[i, :2*i] = 0
        super().__init__(np.ones((1, 1), dtype = int), **params)
        self.content = content*m
        self.active = (self.content//self.limit).any()
        #
        self.mask_master = m
    
    def evo_step(self):
        t0 = self.perf()
        d = self.content//self.limit
        if d.any():
            self.content -= self.limit*d
            self.content[:-1, :] += d[1:, :]
            self.content[1:, :] += d[:-1, :]
            self.content[:, :-1] += d[:, 1:]
            self.content[:, 1:] += d[:, :-1]
            self.content[:-1, :-1] += d[1:, 1:]
            self.content[1:, 1:] += d[:-1, :-1]
            # B
            v = d[0, 1]
            self.content[0, 0] += 5*v
            self.content[0, 1] += 2*v
            self.content[1, 2] += v
            # C, D
            self.content[1, 2] += d[0, 2]
            self.content[0, 2] += d[1, 2]
            self.content[0, 1] += d[1, 2]
            # E
            v = d[1, 3]
            self.content[1, 3] += v
            self.content[1, 2] += v
            self.content[0, 2] += v
            self.content[0, 3] += v
            self.content[2, 4] += v
            # F
            v = d[1, 4]
            self.content[0, 3] += v
            self.content[0, 1] += v
            self.content[2, 4] += v
            # G
            self.content[0, 4:-1] += d[1, 5:]
            self.content[0, 5:] += d[1, 5:]
            # H
            h, w = self.content.shape
            for i in range(2, h-1):
                #print('h', i)
                v = d[i, 1+2*i]
                for j in ((i, 2*i), (i, 1+2*i), (i+1, 2+2*i)):
                    #print('h', j)
                    self.content[j] += v
            if not w%2:
                i = h-1
                #print('pch', i, 1+2*i)
                v = d[i, 1+2*i]
                for j in ((i, 2*i), (i, 1+2*i)):
                    #print('pch', j)
                    self.content[j] += v
            # I
            for i in range(2, h-1):
                self.content[1+i, 2+2*i] += d[i, 2+2*i]
            #
            self.content *= self.mask_master
            s = self.get_plane().sum()
            v = 0
            self.evo_time += self.perf()-t0
            self.steps += 1
        else:
            self.evo_time += self.perf()-t0
            self.active = False

    def get_plane(self): ###
        h, w = self.content.shape
        d = 2*w-1
        c = w-1
        res = np.zeros((d, d), dtype = int)
        res[c:c+h, c:] = self.content
        for x in range(1, w):
            for y in range((x+1)//2):
                res[c+x-y, c+x] = self.content[y, x]
        q = res[c:, c:].copy()
        for i in range(1, w):
            q[i, :] = np.roll(q[i, :], -i)
        res[:c, c:] = q[:0:-1, :]
        q = res[c:, c:].copy()
        for i in range(w):
            res[c+i, c+i] = 0
        res[c:, c:] += q.T
        res[:, :c] = res[::-1, :c:-1]
        #res[m:, :n] = self.content[:, :0:-1]
        #res[:m, n:] = self.content[:0:-1, :]
        #res[:m, :n] = self.content[:0:-1, :0:-1]
        return res
    
    def parse_plane(self, plane): # to do
        pass
        #h, w = plane.shape
        #return plane[h//2:, w//2:]

class X6W6F12(N6W6F12, ExpandableASP):
    def __init__(self, content, **params):
        super().__init__(content, **params)
        h, w = self.content.shape
        if w < 8:
            c = np.zeros((4, 8), dtype = int)
            c[:h, :w] = self.content
            super().__init__(c, **params)
    
    def expand(self):
        t0 = self.perf()
        h, w = self.content.shape
        if self.content[:, -1].any():
            #print('expanding')
            self.content = np.hstack((self.content, np.zeros((h, 1), dtype = int)))
            self.mask_master = np.hstack((self.mask_master, np.ones((h, 1), dtype = int)))
            if not w%2:
                b = np.zeros((1, w+1), dtype = int)
                self.content = np.vstack((self.content, b))
                self.mask_master = np.vstack((self.mask_master, np.hstack((b[:, :-1], np.ones((1, 1), dtype = int)))))
        self.exp_time += self.perf()-t0

class S6W6(StackASP, X6W6F1):
    def __init__(self, height, **params):
        super().__init__(height, **params)
    
    def expand(self): ### Here while folded expansion is not implemented
        t0 = self.perf()
        d = self.content.shape[0]+2
        if self.content[0, :].any():
            c = np.zeros((d, d), dtype = int)
            c[1:-1, 1:-1] = self.content
            self.content = c
            self.mask_master = np.ones((d, d), dtype = int)
            sl = (d+1)//2
            for i in range(sl-1):
                self.mask_master[i, 1-sl+i:] = 0
                self.mask_master[-1-i, :sl-1-i] = 0
        self.exp_time += self.perf()-t0



Imager = ImagerSingleton()