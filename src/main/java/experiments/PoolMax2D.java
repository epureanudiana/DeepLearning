/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;

import nl.tue.s2id90.dl.NN.error.IllegalInput;
import nl.tue.s2id90.dl.NN.initializer.Initializer;
import nl.tue.s2id90.dl.NN.layer.Layer;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author dianaepureanu
 */
public class PoolMax2D extends Layer {
    
    /* Pooling layer constructor */
    public PoolMax2D( String layerName , TensorShape inputShape , 
            int stride);
    @Override
    public boolean showValues() {
        throw new UnsupportedOperationException("Not supported yet."); 
    }

    @Override
    public void initializeLayer(Initializer initializer) {
        throw new UnsupportedOperationException("Not supported yet."); 
    }

    @Override
    public Tensor inference(Tensor input) throws IllegalInput {
        throw new UnsupportedOperationException("Not supported yet."); 
    }

    @Override
    public INDArray backpropagation(INDArray input) {
        throw new UnsupportedOperationException("Not supported yet."); 
    }

    @Override
    public void updateLayer(float learning_rate, int batch_size) {
        throw new UnsupportedOperationException("Not supported yet."); 
    }
    
}
