# Refactor

 refactor the tensor architecture so that it runs faster

 reason of slow: ref to the address for every backprop op

 solution: use self a lot less often

# TODO

 add dropout: restore-release part and voila: done

 add monitor of training also, might be a good practice to wrap up .py into .exe

 custom forward pass, starts from CNN(50%), and implement backprop as well (hint: treat it as convolution)


 for efficiency:

   configure the layers with 'dict', since dict is run by reference, which should save us some time 

   at the update-weight step (where optimizer takes action), instead of redeclare the weights 

   and assign them to the model in the training loop


 Graph

   f(ith): forward pass from layer i, e.g. output is f(last)

  
     input --> layer 1 --> layer 2 --> ... --> layer n --> output
	     					     gradient <-- loss
         d_layer 1 <- d_layer 2 <- ... <- d_layer n
  
  
   To compute d_layer ith: will involve f(i-1).T, diff_act(i-1), layer i+1 ,and gradient

   super hard(100% done): 

    modularize all the stuff to deploy auto differentiation without hand-coded 

    procedures of backprop everytime a new model is establish

    now, it's slow

 Save model

   if one couldn't save the progress, it would evolve slow



