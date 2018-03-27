/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;
import java.io.IOException;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.activation.RELU;
import nl.tue.s2id90.dl.NN.initializer.Gaussian;
import nl.tue.s2id90.dl.NN.layer.Flatten;
import nl.tue.s2id90.dl.NN.layer.FullyConnected;
import nl.tue.s2id90.dl.NN.layer.InputLayer;
import nl.tue.s2id90.dl.NN.layer.OutputSoftmax;
import nl.tue.s2id90.dl.NN.layer.SimpleOutput;
import nl.tue.s2id90.dl.NN.loss.CrossEntropy;
import nl.tue.s2id90.dl.NN.loss.MSE;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.optimizer.SGD;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.NN.validate.Classification;
import nl.tue.s2id90.dl.NN.validate.Regression;
import nl.tue.s2id90.dl.experiment.Experiment;
import nl.tue.s2id90.dl.input.GenerateFunctionData;
import nl.tue.s2id90.dl.input.InputReader;
import nl.tue.s2id90.dl.input.MNISTReader;
import nl.tue.s2id90.dl.javafx.FXGUI;
import nl.tue.s2id90.dl.javafx.ShowCase;

/**
 *
 * @author dianaepureanu
 */
public class ZalandoExperiment extends Experiment {
     // ( hyper ) parameters
    int batchSize = 32;
    int epochs = 5;
    float learningRate = 0.01f;
    String [] labels= {"T shirt/top" ,"Trouser" ,"Pullover" ,"Dress" ,"Coat" ,
    "Sandal" ,"Shirt" ,"Sneaker" ,"Bag" ,"Ankle boot" };
    
    ZalandoExperiment(){ 
        super(true) ; 
    }

    public void go() throws IOException {
    // you are going to add code here.
    // read input and print some information on the data
    InputReader reader = MNISTReader.fashion(batchSize); 
    System.out.println("Reader info:\n" + reader.toString());
    int inputs = reader.getInputShape().getNeuronCount();
    int outputs = reader.getOutputShape().getNeuronCount();
    //print one record
    reader.getValidationData(1).forEach(System.out:: println);
    Model m = createModel(inputs, outputs);
    m.initialize (new Gaussian());

    ShowCase showCase = new ShowCase( i  -> labels[i]); 
    FXGUI.getSingleton().addTab("show case", showCase.getNode()); 
    showCase.setItems(reader.getValidationData(100));
    
    Optimizer sgd = SGD.builder().model(m).learningRate(learningRate)
        .validator(new Classification()).build();
    trainModel(m, reader, sgd, epochs, 0);
    }
    
    Model createModel ( int inputs , int outputs ) {
    Model model = new Model(new InputLayer("In", new TensorShape(inputs), true)); 
    
    // add flatten layer after input layer
    model.addLayer(new Flatten ("Flatten", new TensorShape(16)));
    model.addLayer(new OutputSoftmax("Out",new TensorShape(16), 1, new CrossEntropy()));
   
    //System.out.println(model); 
    return model ;
}
    
    public static void main(String[] args) throws IOException {
        new ZalandoExperiment().go();
   }
}
