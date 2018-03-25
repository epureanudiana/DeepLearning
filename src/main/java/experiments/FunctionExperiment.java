/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;
import nl.tue.s2id90.dl.experiment.Experiment;
import java.io.IOException;
import nl.tue.s2id90.dl.input.GenerateFunctionData;
import nl.tue.s2id90.dl.input.InputReader;

/**
 *
 * @author dianaepureanu
 */
public class FunctionExperiment extends Experiment{
  // ( hyper ) parameters
    int batchSize = 32;
    
    public void go() throws IOException {
    // you are going to add code here.
    // read input and print some information on the data
    InputReader reader = GenerateFunctionData.
            THREE_VALUED_FUNCTION ( batchSize ) ;
    System.out.println("Reader info:\n" + reader.toString());
    reader.getValidationData(10).forEach(System.out::println);
    }
    
    public static void main(String[] args) throws IOException {
        new FunctionExperiment().go();
   }
}
    

