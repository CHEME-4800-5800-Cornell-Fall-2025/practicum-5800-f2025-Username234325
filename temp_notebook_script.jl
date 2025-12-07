# Auto-generated script from notebook
# ---- cell 1 ----
include("Include.jl"); # load a bunch of libs, including the ones we need to work with images

# ---- cell 2 ----
file_extension(file::String) = file[findlast(==('.'), file)+1:end]; # helper function to get the file extension

# ---- cell 3 ----
number_of_training_examples = 30; # how many training examples of *each* number to include from the library
number_digit_array = range(0,length=10,step=1) |> collect; # numbers 0 ... 9
number_of_rows = 28; # number of rows in the image
number_of_cols = 28; # number of cols in the image
number_of_pixels = number_of_rows*number_of_cols; # how many pixels do we have in the image?
number_of_images_to_memorize = 10; # number of images that we want to encode. This is roughly 9% of the theoretical capacity (Kmax ≈ 108), well below the 14% limit to ensure reliable retrieval

# ---- cell 4 ----
training_image_array = let
    
    # initialize -
    training_image_dictionary = Dict{Int64, Array{Gray{N0f8},3}}();
    files = readdir(joinpath(_PATH_TO_IMAGES));
    number_of_files = length(files); 
    image_digit_array = Array{Gray{N0f8},3}(undef, number_of_rows, number_of_cols, number_of_training_examples);

    for i ∈ 1:(number_of_training_examples-1)    
        filename = files[i];
        image_digit_array[:,:,i] = joinpath(_PATH_TO_IMAGES, filename) |> x-> FileIO.load(x);
    end
    image_digit_array;
end;

# ---- cell 5 ----
training_image_dataset = let

    # initialize
    training_image_dataset = Vector{Vector{Float32}}();
    X = training_image_array; # shorthand
    
    for t ∈ 1:number_of_training_examples
        D = Array{Float32,1}(undef, number_of_pixels);
    
        linearindex = 1;
        for row ∈ 1:number_of_rows
            for col ∈ 1:number_of_cols
                D[linearindex] = X[row,col,t] |> x-> convert(Float32,x);
                linearindex+=1;
            end
        end
        push!(training_image_dataset,D);
    end
    training_image_dataset
end;

# ---- cell 6 ----
Kmax = 0.138*number_of_pixels |> x-> round(x, RoundDown) # max number of images the network can memorize

# ---- cell 7 ----
image_index_set_to_encode = let

    # how many images do we want to encode?
    number_of_possible_images = length(training_image_dataset);
    image_index_set_to_encode = Set{Int64}();

    is_ok_to_stop = false; # iteration flag
    while (is_ok_to_stop == false)
        
        # generate a random index -
        j = rand(1:number_of_possible_images);
        push!(image_index_set_to_encode, j); # add to the image set -

        # check: have we hit the number that we want?
        if (length(image_index_set_to_encode) ≥ number_of_images_to_memorize)
            is_ok_to_stop = true;
        end
    end

    # return 
    image_index_set_to_encode;
end;

# ---- cell 8 ----
let

    index_vector = image_index_set_to_encode |> collect |> sort; # we'll process this in this order 
    for example_image_index ∈ index_vector
    
        ŝₖ = training_image_dataset[example_image_index]; # raw state *not* scaled to -1,1
        s = Array{Int32,1}(undef, number_of_pixels); # initialize some space
        for i ∈ 1:number_of_pixels
            pixel =  ŝₖ[i] |> x-> round(Int,x); # why do we have to round here?
            if pixel == 0.0
                s[i] = -1
            else
                s[i] = 1;
            end
        end
        display(decode(s) |> img -> Gray.(img))
    end
end

# ---- cell 9 ----
model = let

    # initialize -
    number_of_images_to_learn = length(image_index_set_to_encode);
    linearimagecollection = Array{Int32,2}(undef, number_of_pixels, number_of_images_to_learn); # images on columns
    
    # turn our set into a sorted vector -
    index_vector = image_index_set_to_encode |> collect |> sort; # we'll process this in this order 
    for k ∈ eachindex(index_vector)
        
        j = index_vector[k];
        ŝₖ = training_image_dataset[j]; # raw state *not* scaled to -1,1

        # fill the columns of the array -
        for i ∈ 1:number_of_pixels
            pixel =  ŝₖ[i] |> x-> round(Int,x);
            if pixel == 0.0 # hmmm
                linearimagecollection[i,k] = -1;
            else
                linearimagecollection[i,k] = 1;
            end
        end
    end

    # build the model using the encode function -
    model = build(MyClassicalHopfieldNetworkModel, (
        memories = linearimagecollection,
    ));

    # return -
    model
end;

# ---- cell 10 ----
imageindextorecover = 1; # which element of the index vector will we choose?

# ---- cell 11 ----
true_image_energy = model.energy[imageindextorecover]

# ---- cell 12 ----
sₒ = let

    # initialize -
    index_vector = image_index_set_to_encode |> collect |> sort; # we'll process this in this order
    index_of_image_to_encode = index_vector[imageindextorecover]; # -or- choose random
    ŝₖ = training_image_dataset[index_of_image_to_encode]; # raw state *not* scaled to -1,1
    sₒ = Array{Int32,1}(undef, number_of_pixels); # initialize some space
    θ = 0.68; # fraction of pixels that we want to keep correct. This means ~32% corruption, providing a challenging but not impossible recovery task

    # let's build the corrupted initial condition -
    for i ∈ 1:number_of_pixels
        pixel =  ŝₖ[i] |> x-> round(Int,x); # We have some gray-scale values in the original vector, need to round
        if pixel == 0.0
            sₒ[i] = -1;
        else
            sₒ[i] = 1;
        end
    end

    # # cut 1 - θ fraction of the pixels, replacewith - 1
    number_of_pixels_to_corrupt = round(Int, (1 - θ)*number_of_pixels);
    start =(number_of_pixels - number_of_pixels_to_corrupt) + 1;
    for i ∈ start:number_of_pixels
        sₒ[i] = -1;
    end

    sₒ # return
end;

# ---- cell 13 ----
decode(sₒ) |> img -> Gray.(img) # corrupted true image. This is what we give the network

# ---- cell 14 ----
frames, energydictionary = recover(model, sₒ, true_image_energy, maxiterations=25*number_of_pixels, 
    patience = number_of_pixels);

# ---- cell 15 ----
let

    # initialize -
    index_vector = image_index_set_to_encode |> collect |> sort; # we'll process this in this order
    my_index_of_image_to_encode = index_vector[imageindextorecover]; # -or- choose random

    # true image -
    ŝₖ = training_image_dataset[my_index_of_image_to_encode]; # raw state *not* scaled to -1,1
    s₁ = Array{Int32,1}(undef, number_of_pixels); # initialize some space
    for i ∈ 1:number_of_pixels
        pixel =  ŝₖ[i] |> x-> round(Int,x); # why do we have to round here?
        if pixel == 0.0
            s₁[i] = -1
        else
            s₁[i] = 1;
        end
    end
    true_image = decode(s₁); # this is the true image
    initial_image = decode(sₒ); # initial corrupted image
    
    # recovered image -
    ks = collect(keys(energydictionary))
    best_key = argmin(k -> energydictionary[k], ks)
    best_state = frames[best_key]
    recovered_image = decode(best_state)

    display(true_image |> img -> Gray.(img))
    display(initial_image |> img -> Gray.(img))
    display(recovered_image |> img -> Gray.(img))

    println("Hamming (best vs true) = ", hamming(best_state, s₁))
    println("Hamming (initial vs true) = ", hamming(sₒ, s₁))
    println("Best energy = ", energydictionary[best_key])
    println("True energy = ", true_image_energy)

    # check: the hamming distance between the best and true should be less than that between the initial and true
    @assert hamming(best_state, s₁) < hamming(sₒ, s₁) "Error: Hamming distance check failed!"
end;

# ---- cell 16 ----
let
    p = plot(bg="gray95", background_color_outside="white", framestyle = :box, fg_legend = :transparent); 
    plot!(energydictionary, lw=3, c=:navy, label="", xminorticks=true, yminorticks=true);
   
    # plot true energy line -
    TEL = true_image_energy*ones(length(energydictionary));
    plot!(TEL, lw=2, c=:red, label="True image energy", ls=:dash);
    xlabel!("Step index (AU)", fontsize=18)
    ylabel!("Network configuration energy (AU)", fontsize=18)
end

# ---- cell 17 ----
@testset verbose = true "CHEME 5800 Practicum Test Suite" begin

    @testset "Setup, Data, and Prerequisites" begin
        # Test basic constants
        @test number_of_pixels == number_of_rows * number_of_cols
        @test Kmax ≈ 0.138 * number_of_pixels atol=1  # approximate due to rounding
        @test number_of_training_examples > 0
        @test length(number_digit_array) > 0  # should have some digits
        @test 0 < number_of_images_to_memorize <= Kmax
        
        # Test data loading
        @test !isempty(training_image_dataset)
    end

    @testset "Task 1: Learn the Weights of the Network" begin
        # Test image selection
        @test length(image_index_set_to_encode) == number_of_images_to_memorize
        @test isa(image_index_set_to_encode, Set)
        @test all(1 .<= collect(image_index_set_to_encode) .<= length(training_image_dataset))
        
        # Test model properties
        @test size(model.W) == (number_of_pixels, number_of_pixels)
        @test model.W ≈ model.W'  # weight matrix should be symmetric
        @test all(iszero, model.b)  # bias should be zero for classical Hopfield
        @test all(iszero, diag(model.W))  # no self-connections
        @test length(model.energy) == number_of_images_to_memorize
    end

    @testset "Task 2: Retrieve a Memory from the Network" begin
        # Test recovery process
        energy_keys = collect(keys(energydictionary))
        initial_key = minimum(energy_keys)
        initial_energy = energydictionary[initial_key]
        final_energy = minimum(values(energydictionary))
        @test final_energy <= initial_energy  # energy should not increase
        @test length(frames) > 1  # should have run some iterations
        @test keys(frames) == keys(energydictionary)  # same keys
        
        # Test best state - all states should be binary (from Hopfield network)
        best_key = argmin(k -> energydictionary[k], energy_keys)
        best_state = frames[best_key]
        @test all(s -> abs(s) == 1, best_state)  # binary states (either +1 or -1)
    end

end;

