import torch
from checkpoint import save_checkpoint

def train(model, optimizer, criterion, trainloader, validationloader, device, epochs=2):
    print(optimizer)
    print(criterion)
    
    t_losses = []
    v_losses = []
    num_epochs = epochs
    for epoch in range(num_epochs):

        model.train()
        running_loss = 0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            def closure():
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                return loss

            loss = optimizer.step(closure)

            print(f'Epoch {epoch+1}, Step {i}, Loss: {loss}')
            running_loss += loss.item()

            del inputs, labels

        t_losses.append(running_loss / len(trainloader))
        print(f"Epoch {epoch+1} end, avg train loss: {running_loss / len(trainloader)}")

        model.eval()  # Set model to evaluation mode
        val_running_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():  # No need to track gradients during validation
            for v_i, (inputs, labels) in enumerate(validationloader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                del inputs, labels, outputs
        
        # Calculate and store validation loss and accuracy
        val_loss = val_running_loss / len(validationloader)
        v_losses.append(val_loss)
        accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1} end, avg val loss: {val_loss}, accuracy: {accuracy:.2f}%")

        if (epoch % 10 == 0 and epoch != 0):
          save_checkpoint(model, optimizer, epoch+1, loss=val_loss, accuracy=accuracy, filename=f"checkpoint_{epoch+1}.pth", t_losses=t_losses, v_losses=v_losses)


    model = model.cpu()
    model.t_losses = t_losses
    model.v_losses = v_losses
    torch.cuda.empty_cache()
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")



    print("Training complete.")
    return model