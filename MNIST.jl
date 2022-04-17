using Flux, MLDatasets, CUDA, FileIO
using Flux: train!, onehotbatch
x_train, y_train = MLDatasets.MNIST.traindata()
x_test, y_test = MLDatasets.MNIST.testdata()

x_train = Float32.(x_train)
y_train = Flux.onehotbatch(y_train, 0:9)
model = Chain(
    Dense(784, 256, relu),
    Dense(256, 10, relu), softmax
)

loss(x, y) = Flux.Losses.logitcrossentropy(model(x), y)
optimizer = ADAM(0.0001)
parameters = params(model)
train_data = [(Flux.flatten(x_train), y_train)]
test_data = [(Flux.flatten(x_test), y_test)]

for i in 1:400
    Flux.train!(loss, parameters, train_data, optimizer)
end

check = println((findmax(model(train_data[1][1][:, 1]))) .+ (0, -1), y_train[1])

check_test = println(findmax(model(Float32.(Flux.flatten(x_test))[:, 1])) .+ (0, -1), y_test[1])

accuracy = 0
for i in 1:length(y_test)
    if findmax(model(test_data[1][1][:, i]))[2] - 1  == y_test[i]
        accuracy = accuracy + 1
    end
end

println(accuracy / length(y_test))
