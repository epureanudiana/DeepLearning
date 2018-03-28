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
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author dianaepureanu
 */
public class GradientDescentWithMomentum implements UpdateFunction {
    
  Optimizer sgd = SGD.builder().
          updateFunction(GradientDescentWithMomentum ::new).build() ;
    
    INDArray velocity = Nd4j.zeros(28,28,1);
    double mu = 0.9; 
// coefficient of friction
    @Override
    /* Does a gradient descent step with factor ’minus learningRate’ and corrected for batchSize. */
    public void update(INDArray array, boolean isBias, float learningRate, 
            int batchSize, INDArray gradient) {
        
       float factor = (learningRate/batchSize ) ;
       Nd4j.getBlasWrapper().level1().axpy( array.length(), factor, gradient, array );
       gradient.assign(0); // array <   array + factor * gradient
       
       velocity = velocity.mul(mu).sub(gradient.mul(learningRate));
       array.addi(velocity);
    }
}
