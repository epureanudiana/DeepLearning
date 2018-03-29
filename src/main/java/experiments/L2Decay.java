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

/**
 *
 * @author Veronika
 */
public class L2Decay implements UpdateFunction {
    
    INDArray update;
    int currentEpoch;
    private float alpha;
    UpdateFunction f ;
    
    public L2Decay ( Supplier <UpdateFunction> supplier , float aplha ) {
        this.alpha = alpha ;
        this.f = supplier.get() ;
    }
    
    @Override
    public void update(INDArray array, boolean isBias, float learningRate, int batchSize, INDArray gradient) {
        
        f.update(array, isBias, learningRate, batchSize, gradient);
        if (isBias){
            return;
        } else{
            
         Nd4j.getBlasWrapper().level1().axpy(array.length(), -alpha, array, array);
         
        }
        
        
    }
    
    
    public float getAlpha(){
        return this.alpha;
    }
    public void setAlpha(float newAlpha){
        this.alpha = newAlpha; 
    }
    
    public void setEpoch(int e){
        this.currentEpoch = e; 
    }
}


