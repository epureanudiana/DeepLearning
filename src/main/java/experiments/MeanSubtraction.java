/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;

import java.util.List;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.NN.transform.DataTransform;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author dianaepureanu
 */
public class MeanSubtraction implements DataTransform { 
    
Float mean;

@Override
/** computes statistics for the dataset consisting of input output pairs, these statistics
* are used in the transform method. Note that the stats are only based on the input.
* @param pairs dataset **/
public void fit( List <TensorPair > data ) { 
    if (data.isEmpty()) {
        throw new IllegalArgumentException("Empty dataset") ;
    }
    float sum = 0;
    for(TensorPair pair : data) {
        //data is stored in an INDArray.
        INDArray a = pair.model_input.getValues();
//        if (a.rank() == 1){
//            System.out.println("Yes, this array has only one dimension!");
//        } else {
//            System.out.println("Sorry, no " + a.shapeInfoToString() + " " + a.shape());
//        }
//        System.out.println(a.shapeInfoToString() + " " +'\n' + a.getFloat(0) + " " +a.getFloat(1)
//        + " " +a.getFloat(2)+ " " + a.getFloat(3));
      //  System.out.println(a);
        sum += a.meanNumber().floatValue();
    }
    mean = sum/(float)data.size();
}

@Override
/** transforms the dataset, using the statistics calculated by the fit method. 
 * @param pairs dataset **/
 public void transform(List<TensorPair> data) {
      for (TensorPair pair : data) {
          pair.model_input.getValues().subi(mean);
      }
    }
 }