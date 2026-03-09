1) Tensors - they are the arrays very similar to numpy arrays with the only difference that they can run on GPU and other high computation hardware

2) How to use a pre-trained resnet18 architecture model with pre-trained weights to see how pytorch learns the parameters 

3) torch.autograd - it is a mechanism in pytorch where we don't have to define how to calculate the derivatives, but the autograd will itself do it.
what we do from our side :-
- forward pass
- calculate error
- then we do backward pass where automatically the autograd will be triggered to calculate the gradients for all the parameters in the network and will be accordingly updated.
- for each batch, once this process is done then we do reset to make sure that the gradients from the previous batch are not accumultaed in the current batch

4) in the first example ( Intro_pytorch_autograd.ipynb), we took a pre trained model architecture resnet18 with the pre learnt parameters and then we try different steps  - 
define the data
model ( which will be the pre selected model)
forward pass - which will generate the predictions
calculate the error/loss
backward pass - this will trigger the autograd to update the gradients ( differentiation) for each parameter in the network. 
once the gradients are ready, then we call the optimizer to update the parameters.

5) Question - When you called loss.backward() on the ResNet18 example — how did PyTorch know how to compute the gradient for every single weight in that massive network without you writing a single derivative formula?
Answer - when we call loss.backward() then atomatically autograd is triggered which is automatic gradient. it has the capability to calculate the gradients  on the loss w.r.t the parameters for the entire network. for each batch , it will calculate the gradients and store them in the .grad variables and once we update the params, then we need to zero_grad to make sure that the graident from previous batch are not aggregated in the current batch

6) Question - How autograd learns the derivatives without we explicitly writing?
Answer - Using computational graph:
 When you write this:
y_pred = w * x + b
loss = ((y_pred - y) ** 2).mean()

# PyTorch silently builds a graph of every operation:
x → multiply by w → add b → subtract y → square → mean → loss

# loss.backward() walks this graph BACKWARDS
# applying chain rule at each operation automatically
# This is why it's called backPROPAGATION —
# it propagates gradients backwards through the graph

7) Question - During your ResNet18 experiment — what happened when you called loss.backward() on a tensor that had requires_grad=False? Did you notice anything? And why does requires_grad exist at all?
Think about it in context of fine-tuning — if you're fine-tuning only the last layer of ResNet18 and freezing everything else, which layers would have requires_grad=True and which would have requires_grad=False? 
Answer - for the ones where requires_grad = False, autograd will not store their respective gradients. requires_grad exist to only store the gradients for the parameters we think should be updated and the ones which are expected to remain constant can be set to requires_grad = False

