/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;

import java.util.function.Supplier;
import nl.tue.s2id90.dl.NN.optimizer.update.UpdateFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.ops.transforms.Transforms;
import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;

/**
 *
 * @author dianaepureanu
 */
public class MyAdaDelta extends AdaDelta implements UpdateFunction {
    
    INDArray meanGradient;
    INDArray meanDelta;
    INDArray rmsGradient;
    INDArray rmsDelta;
    INDArray delta;
    double rho;
    double epsilon;
    UpdateFunction l2Decay;
    
    public MyAdaDelta(Supplier<UpdateFunction> supplierDecay) {

        this.rho = AdaDelta.DEFAULT_ADADELTA_RHO;
        this.epsilon = AdaDelta.DEFAULT_ADADELTA_EPSILON;

        //be able to do moment and L2decay
        l2Decay = supplierDecay.get();

    }
   
    @Override
    public void update(INDArray array, boolean isBias, float learningRate, 
            int batchSize, INDArray gradient){
        
        l2Decay.update(array, isBias, learningRate, batchSize, gradient);
        
        if (meanGradient == null) {
            meanGradient = array.dup('f').assign(0);
        }
        if (meanDelta == null) {
            meanDelta = array.dup('f').assign(0);
        }
        if (rmsGradient == null) {
            rmsGradient = array.dup('c').assign(1);
        }
        if (rmsDelta == null) {
            rmsDelta = array.dup('c').assign(1);
        }
        
    //Accumulate gradient    
    //currentMeanGradient = rho * previousMeanGradient + (1-rho) * (currentGradient^2)
    meanGradient.muli(rho).addi(gradient.mul(gradient).muli(1 - rho));
    //Compute rmsGradient
    //RMS[meanGradient] = sqrt(meanGradient + epsilon)
    rmsDelta.muli(sqrt(meanDelta.addi(epsilon)));
    rmsGradient.muli(sqrt(meanGradient.addi(epsilon)));
    //Compute update
    // update = -gradient * (RMS[meanDelta]/RMS[meanGradient])     
    //update is already -1!!
    // delta.muli((meanDelta.divi(rmsGradient)).muli(gradient));
    delta = gradient.muli(rmsDelta.divi(rmsGradient));
    //Accumulate update
    // RMS[meanDelta] = (RMS[meanDelta]* rho) + (1-rho)*(update^2)  
    meanDelta.muli(rho).addi(delta.mul(delta).muli(1-rho));  
    //Apply update
    //Nd4j.getBlasWrapper().level1().axpy(array.length(), 1, delta, array);
    Nd4j.getBlasWrapper().level1().axpy(gradient.length(), 1 , meanDelta, gradient);     
    }
}

