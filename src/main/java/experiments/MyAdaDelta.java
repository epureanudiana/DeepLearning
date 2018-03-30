/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;

import nl.tue.s2id90.dl.NN.optimizer.update.UpdateFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaDelta;
import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;

/**
 *
 * @author dianaepureanu
 */
public class MyAdaDelta extends AdaDelta implements UpdateFunction {
    
    INDArray currentMeanGradient;
    INDArray previousMeanGradient;
    INDArray delta;
    
//    public MyAdaDelta(){
//        super();
//    }
    @Override
    public void update(INDArray array, boolean isBias, float learningRate, 
            int batchSize, INDArray gradient){
        
        if (previousMeanGradient == null) {
            previousMeanGradient = array.dup('f').assign(0);
        }
        double epsilon = this.getEpsilon();
        double decay = this.getRho();
        
        currentMeanGradient = previousMeanGradient.muli(decay).
                addi(gradient.muli(1 - decay));
        previousMeanGradient = currentMeanGradient;
        delta = (sqrt(previousMeanGradient.addi(epsilon)).
                divi(sqrt(currentMeanGradient.addi(epsilon)))).muli(gradient);
        
        delta.muli(-1);
        
        Nd4j.getBlasWrapper().level1().axpy(array.length(), 1, delta, array);
        
    }
}
