/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;
import nl.tue.s2id90.dl.experiment.Experiment;
import java.io.IOException;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.initializer.Gaussian;
import nl.tue.s2id90.dl.NN.layer.InputLayer;
import nl.tue.s2id90.dl.NN.layer.SimpleOutput;
import nl.tue.s2id90.dl.NN.loss.Loss;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.optimizer.SGD;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.NN.validate.Regression;
import nl.tue.s2id90.dl.input.GenerateFunctionData;
import nl.tue.s2id90.dl.input.InputReader;

/**
 *
 * @author dianaepureanu
 */
public class FunctionExperiment extends Experiment{
  // ( hyper ) parameters
    int batchSize = 32;
    int epochs = 10;
    float learningRate = 0.01f;
    
    public void go() throws IOException {
    // you are going to add code here.
    // read input and print some information on the data
    InputReader reader = GenerateFunctionData.
            THREE_VALUED_FUNCTION ( batchSize ) ;
    System.out.println("Reader info:\n" + reader.toString());
    int inputs = reader.getInputShape().getNeuronCount();
    int outputs = reader.getOutputShape().getNeuronCount();
    reader.getValidationData(10).forEach(System.out::println);
    Model m = createModel(inputs, outputs);
    System.out.print(m);
    m.initialize (new Gaussian());
    // Training : create and configure SGD && train model
    Optimizer sgd = SGD.builder(). model( m ).validator(new Regression()).
            learningRate(learningRate).build() ;
    trainModel(m, reader, sgd, epochs, 0);
    }
    
    Model createModel ( int inputs , int outputs ) {
    Model model = new Model(new InputLayer("In", new TensorShape(inputs), true)); 
    model.addLayer(new SimpleOutput("Out", new TensorShape(inputs), outputs, newMSE(), true));
    System.out.println(model); 
    return model ;
}
    
    public static void main(String[] args) throws IOException {
        new FunctionExperiment().go();
   }

    private Loss newMSE() {
       return null;
    }
}
    

