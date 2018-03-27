/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;

import java.util.List;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.NN.transform.DataTransform;

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
    for(TensorPair pair : data) {
    }
}

@Override
/** transforms the dataset, using the statistics calculated by the fit method. 
 * @param pairs dataset **/
 public void transform(List<TensorPair> data) {
      
    }
}