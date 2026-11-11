import io,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
import runtime as R
import protocol as P

class Contracts(unittest.TestCase):
    def setUp(self):
        self.ids=np.asarray(['b','a','d','c']);self.g=R.unit(np.eye(4,dtype=np.float32));self.q=self.g[:,0]
    def test_qe_guards_and_ties(self):
        v,_=R.qe(self.q,self.g,self.ids,lam=0);np.testing.assert_array_equal(v,self.q)
        v,d=R.qe(self.q,self.g,self.ids,tau=1.1);np.testing.assert_array_equal(v,self.q);self.assertFalse(d['open'])
        np.testing.assert_array_equal(R.ranking(np.zeros((4,1)),self.ids)[:,0],[1,0,3,2])
    def test_fusion(self):
        s=np.asarray([[.3],[.8],[.2],[.4]])
        np.testing.assert_array_equal(R.ranking(R.fuse([s],self.ids),self.ids),R.ranking(s,self.ids))
        np.testing.assert_allclose(R.fuse([s,s],self.ids,'average'),s)
    def test_graph(self):
        g=R.unit(np.random.default_rng(1).normal(size=(8,12)).astype(np.float32));ids=np.array([str(i) for i in range(12)])
        for mutual in [False,True]:
            graph,stats=R.build_graph(g,ids,3,mutual);self.assertTrue(np.all(graph.diagonal()==0));self.assertTrue(np.allclose(graph.toarray(),graph.T.toarray()))
            scores=g.T@g[:,:2];v,di=R.diffuse(scores,graph,beta=0)
            expected=np.maximum(scores,0).astype(float)**3;expected/=expected.sum(0)
            np.testing.assert_allclose(v,expected);self.assertTrue(all(di['converged']))
            v,di=R.diffuse(-np.ones((12,1)),graph);self.assertTrue(di['fallback'][0]);np.testing.assert_equal(v,-1)
    def test_metric_junk(self):
        gt=dict(gnd=[dict(ok=[1],junk=[0])]);m,rows=R.measure(np.array([[0],[1],[2]]),gt);self.assertEqual(m,1);self.assertEqual(rows[0]['recall20'],1)
        m,_=R.measure(np.array([[2],[0],[1]]),gt);self.assertLess(m,1)
    def test_rerank(self):
        s=np.array([.8,.7,.6,.5]);r,f=R.rerank(s,self.ids,{},k=2);self.assertTrue(f);np.testing.assert_equal(r,[0,1,2,3])
        evidence={0:dict(matches=20,inliers=0,valid=False),1:dict(matches=30,inliers=25,valid=True)}
        r,f=R.rerank(s,self.ids,evidence,k=2);self.assertFalse(f);np.testing.assert_equal(r,[1,0,2,3])
    def test_corrupt_and_features(self):
        image=Image.fromarray(np.random.default_rng(10).integers(0,256,(96,128,3),dtype=np.uint8))
        self.assertEqual(len(P.conditions()),31)
        for c in P.conditions():self.assertEqual(R.pixel_hash(R.corrupt(image,c,'q')),R.pixel_hash(R.corrupt(image,c,'q')))
        direct=R.local_features(image);packed,size,h=R.feature_unpack(R.feature_pack(image));np.testing.assert_array_equal(direct[0],packed[0]);np.testing.assert_array_equal(direct[1],packed[1]);self.assertEqual(h,R.pixel_hash(image))
        self.assertFalse(R.geometry((np.zeros((0,2)),np.zeros((0,128))),(np.zeros((0,2)),np.zeros((0,128))),(128,96))['valid'])
    def test_chunk_transport(self):
        from artifacts_io import pack,unpack
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'source';root.mkdir();(root/'results').mkdir()
            payload=bytes(range(255))*100;(root/'results'/'fixture.bin').write_bytes(payload)
            manifest=pack(root,4096);self.assertGreater(len(manifest['parts']),1)
            unpack(root/'artifact_manifest.json',Path(d)/'restored')
            self.assertEqual((Path(d)/'restored'/'results'/'fixture.bin').read_bytes(),payload)
            self.assertFalse((root/'results'/'fixture.bin').exists())

    def test_writer_subset_and_idempotence(self):
        with tempfile.TemporaryDirectory() as p:
            writer=R.Writer(p,dict(models=[dict(id='m')]),'test');gt=dict(qimlist=['q'],gnd=[dict(ok=[1],junk=[0])]);ids=np.array(['x','y','z'])
            for _ in range(2):writer.emit('tiny',gt,'m',P.conditions()[0],'cosine',{},ids,scores=np.array([[.9],[.8],[.1]]),latencies=[1.])
            self.assertEqual(len(list(Path(p).rglob('*.json'))),1)

if __name__=='__main__':unittest.main()
