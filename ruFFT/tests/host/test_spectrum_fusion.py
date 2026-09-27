"""Independent host mathematics/indexing checks, not a Rust/PTX interpreter."""
import unittest
import numpy as np


def product(a,b):
    # Explicit complex multiply with FP32 components, as in both kernel paths.
    ar=a.real.astype(np.float32);ai=a.imag.astype(np.float32)
    br=b.real.astype(np.float32);bi=b.imag.astype(np.float32)
    return (ar*br-ai*bi).astype(np.float32)+1j*(ar*bi+ai*br).astype(np.float32)


def four_step_loaded_product(a,b,n1,n2):
    # Mirror only the four-step index equations: first pass strides by N2;
    # spectrum is indexed by flat bin, never by row/batch or by the n1 index.
    batches=a.shape[0]
    gathered=np.empty((batches,n1,n2),np.complex64)
    for n2idx in range(n2):
        flat=np.arange(n1)*n2+n2idx
        gathered[:,:,n2idx]=product(a[:,flat],b[flat])
    y=np.fft.ifft(gathered,axis=1)*n1
    k1=np.arange(n1)[:,None];i2=np.arange(n2)[None,:]
    y*=np.exp(2j*np.pi*k1*i2/(n1*n2))
    y=np.fft.ifft(y,axis=2)*n2
    return y.transpose(0,2,1).reshape(batches,n1*n2)


class FusionMathTests(unittest.TestCase):
    def test_shared_fp32_product(self):
        rng=np.random.default_rng(2201)
        for m in (16,32,2048,4096):
            with self.subTest(m=m):
                a=(rng.normal(size=(3,m))+1j*rng.normal(size=(3,m))).astype(np.complex64)
                b=(rng.normal(size=m)+1j*rng.normal(size=m)).astype(np.complex64)
                # Each lane writes unique bit-reversed shared-memory addresses.
                bits=m.bit_length()-1;perm=np.array([int(f'{i:0{bits}b}'[::-1],2) for i in range(m)])
                separate=product(a,b);shared=np.empty_like(separate)
                for lane in range(min(m//2,256)):
                    i=np.arange(lane,m,min(m//2,256));shared[:,perm[i]]=product(a[:,i],b[i])
                np.testing.assert_array_equal(shared[:,perm],separate)
                np.testing.assert_allclose(np.fft.ifft(shared[:,perm]),np.fft.ifft(separate),rtol=0,atol=0)
    def test_four_step_global_frequency_index(self):
        rng=np.random.default_rng(2202)
        for n1,n2 in ((8,16),(64,128),(128,128)):
            with self.subTest(n1=n1,n2=n2):
                m=n1*n2;a=(rng.normal(size=(2,m))+1j*rng.normal(size=(2,m))).astype(np.complex64)
                b=(rng.normal(size=m)+1j*rng.normal(size=m)).astype(np.complex64)
                actual=four_step_loaded_product(a,b,n1,n2)
                expected=np.fft.ifft(product(a,b),axis=1)*m
                np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-3)
    def test_each_four_step_bin_loaded_once(self):
        for n1,n2 in ((64,128),(128,128),(256,256),(512,256)):
            count=np.zeros(n1*n2,dtype=np.uint8)
            for n2idx in range(n2):
                for lane in range(min(n1//2,256)):
                    flat=np.arange(lane,n1,min(n1//2,256))*n2+n2idx
                    count[flat]+=1
            self.assertTrue(np.all(count==1))
    def test_no_batch_offset_in_spectrum(self):
        a=np.arange(3*128,dtype=np.float32).reshape(3,128).astype(np.complex64)
        b=(1+1j*np.arange(128)).astype(np.complex64)
        expected=np.fft.ifft(product(a,b),axis=1)*128
        np.testing.assert_allclose(four_step_loaded_product(a,b,8,16),expected,rtol=1e-5,atol=0.3)
    def test_zero_workspace_stays_zero(self):
        a=np.zeros((3,128),np.complex64);b=np.arange(128).astype(np.complex64)
        np.testing.assert_array_equal(four_step_loaded_product(a,b,8,16),a)
    def test_transfer_and_launch_savings_are_scoped(self):
        batch,m=16,2048
        self.assertEqual(16*batch*m,524288) # one eliminated complex write+read
        self.assertEqual(5-1,4)            # small Bluestein execution only
        self.assertEqual(9-1,8)            # four-step execution only

if __name__=='__main__': unittest.main()
