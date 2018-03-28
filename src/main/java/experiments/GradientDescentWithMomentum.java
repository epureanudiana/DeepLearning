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
    
    INDArray update;
    double mu = 0.7; 

    @Override
    public void update(INDArray array, boolean isBias, float learningRate, 
            int batchSize, INDArray gradient) {
    if (update == null) {
       update = array.dup('f').assign(0);
    }    
    //Nd4j.getBlasWrapper().level1().axpy( value.length(), factor, gradient, value );
    // value <-- value + factor * gradient
 
    update.muli(mu);
    Nd4j.getBlasWrapper().level1().axpy(update.length(), -learningRate, gradient, update);  
    Nd4j.getBlasWrapper().level1().axpy(array.length(), 1, update, array);
      // array <-- array + 1*velocity
        
       
    }
}
