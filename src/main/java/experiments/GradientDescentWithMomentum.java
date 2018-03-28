/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;

import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.optimizer.SGD;
import nl.tue.s2id90.dl.NN.optimizer.update.UpdateFunction;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author dianaepureanu
 */
public class GradientDescentWithMomentum implements UpdateFunction {
    
  Optimizer sgd = SGD.builder().
          updateFunction(MyGradientDescentVariant ::new).build() ;

    @Override
    public void update(INDArray array, boolean isBias, float learningRate, 
            int batchSize, INDArray gradient) {
        throw new UnsupportedOperationException("Not supported yet.");
    }
}
