
Throughout the user's guide, we'll use the [Lalonde dataset](http://users.nber.org/~rdehejia/nswdata2.html). It is originally from
Robert Lalonde, "Evaluating the Econometric Evaluations of Training Programs," *American Economic Review*, Vol. 76, pp. 604-620. 1986.

#Getting the data set

To download the example data set, run


    include(joinpath(Pkg.dir("TargetedLearning"), "examples", "fetchdata.jl"))

    INFO: /home/sam/.julia/v0.3/TargetedLearning/examples/data/lalonde.csv exists. To redownload, call fetchdata(true)


This only needs to be run once.

# Loading the data set

## Using readcsv

The simplest way to read the data set is with `readcsv`. The CSV file we have has a header, so we'll pass `header=true`, and we'll get back a tuple containing a matrix with the numeric data in it (`readcsv` can figure out that it's all `Float64`s automatically) and a matrix with one row containing column names.


    dsfname = joinpath(Pkg.dir("TargetedLearning"), "examples", "data", "lalonde.csv")
    
    dat, colnames = readcsv(dsfname, header=true)
    
    #lets inspect colnames
    colnames




    1x9 Array{String,2}:
     "treatment"  "age"  "education"  "black"  …  "nodegree"  "RE75"  "RE78"




    #convert colnames to a vector instead of a matrix with one row
    colnames = reshape(colnames, size(colnames, 2))




    9-element Array{String,1}:
     "treatment"
     "age"      
     "education"
     "black"    
     "hispanic" 
     "married"  
     "nodegree" 
     "RE75"     
     "RE78"     



`treatment` is obviously the treatment variable. The outcome variable is `RE78` (earnings in 1978), and `RE75` is the pretreatment value at baseline. The others are potential baseline confounders. Check the link above for more information about the dataset.  We want to slice the matrix `dat` up to extract the treatment and outcome variable. Julia uses square brackets for indexing. The first dimension of a matrix is rows and the second is columns like R. If you want everything in one dimension, you put `:` (where as in R you can leave that dimension empty). You can index with booleans, integers, vectors of integers, ranges and some other things. Here are some examples:


    #we know column 1 is treatment, so get it directly, and ask for all rows with :
    treatment = dat[:, 1]




    722-element Array{Float64,1}:
     1.0
     1.0
     1.0
     1.0
     1.0
     1.0
     1.0
     1.0
     1.0
     1.0
     1.0
     1.0
     1.0
     ⋮  
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0




    #suppose instead we want to find the position in colnames that has the value
    #"treatment", but we don't know it's the first one. There are a couple of ways
    #to do that.
    #(.== and operators starting with `.` in general indicate that we want to do
    #   an element wise operation)
    
    #(the ; at the end of the line suppresses output)
    treatment_take2 = dat[:, colnames .== "treatment", 1];


    #the last column is the outcome so we can use the keyword `end`
    outcome = dat[:, end]
    
    #we can also use `end` in arithmetic, e.g.
    outcome_prebaseline = dat[:, end-1]




    722-element Array{Float64,1}:
        0.0 
        0.0 
        0.0 
        0.0 
        0.0 
        0.0 
        0.0 
        0.0 
        0.0 
        0.0 
        0.0 
        0.0 
        0.0 
        ⋮   
        0.0 
        0.0 
     1537.93
        0.0 
     8012.09
      768.81
     1345.91
      825.23
        0.0 
     1206.44
        0.0 
        0.0 




    #baseline covariates are everything between the first and last column, so we could use a range:
    baseline_covars = dat[:, 2:end-1]




    722x7 Array{Float64,2}:
     37.0  11.0  1.0  0.0  1.0  1.0     0.0 
     22.0   9.0  0.0  1.0  0.0  1.0     0.0 
     30.0  12.0  1.0  0.0  0.0  0.0     0.0 
     27.0  11.0  1.0  0.0  0.0  1.0     0.0 
     33.0   8.0  1.0  0.0  0.0  1.0     0.0 
     22.0   9.0  1.0  0.0  0.0  1.0     0.0 
     23.0  12.0  1.0  0.0  0.0  0.0     0.0 
     32.0  11.0  1.0  0.0  0.0  1.0     0.0 
     22.0  16.0  1.0  0.0  0.0  0.0     0.0 
     33.0  12.0  0.0  0.0  1.0  0.0     0.0 
     19.0   9.0  1.0  0.0  0.0  1.0     0.0 
     21.0  13.0  1.0  0.0  0.0  0.0     0.0 
     18.0   8.0  1.0  0.0  0.0  1.0     0.0 
      ⋮                         ⋮           
     17.0  10.0  0.0  1.0  0.0  1.0     0.0 
     19.0  11.0  1.0  0.0  0.0  1.0     0.0 
     19.0   9.0  1.0  0.0  0.0  1.0  1537.93
     17.0  10.0  1.0  0.0  0.0  1.0     0.0 
     20.0  11.0  1.0  0.0  0.0  1.0  8012.09
     17.0   8.0  1.0  0.0  0.0  1.0   768.81
     18.0   8.0  0.0  0.0  0.0  1.0  1345.91
     20.0  10.0  1.0  0.0  0.0  1.0   825.23
     17.0   9.0  0.0  1.0  0.0  1.0     0.0 
     17.0   9.0  1.0  0.0  0.0  1.0  1206.44
     19.0  11.0  1.0  0.0  0.0  1.0     0.0 
     18.0  10.0  1.0  0.0  0.0  1.0     0.0 



## Using DataFrame

The `readtable` function returns a `DataFrame` which allows for indexing by column names.


    using DataFrames #`using` is like `library` in R
    
    dsfname = joinpath(Pkg.dir("TargetedLearning"), "examples", "data", "lalonde.csv")
    "/home/sam/.julia/v0.3/TargetedLearning/examples/data/lalonde.csv"
    
    df = readtable(dsfname)
    
    #check the column names:
    names(df)




    9-element Array{Symbol,1}:
     :treatment
     :age      
     :education
     :black    
     :hispanic 
     :married  
     :nodegree 
     :RE75     
     :RE78     



Data frames in Julia are indexed by symbols instead of strings. Symbols are no entirely unlike strings, and are created in julia with `:symbolname` or `Symbol("symbolname")`.

Now let's get the treatment and outcome variables out of `df`.


    treatment = df[:treatment]
    outcome = df[:RE78]




    722-element DataArray{Float64,1}:
      9930.05 
      3595.89 
     24909.5  
      7506.15 
       289.79 
      4056.49 
         0.0  
      8472.16 
      2164.02 
     12418.1  
      8173.91 
     17094.6  
         0.0  
         ⋮    
      8469.27 
         0.0  
      4188.74 
      1143.39 
     18155.8  
      3020.95 
      1725.59 
         0.0  
      5114.81 
         0.0  
         0.0  
       781.224



We see that variables in a data frame are actually `DataArray`s instead of regular Julia `Array`s. The functions in TargetedLearning.jl currently only work with `Arrays` of floating point numbers, so we'll convert them.



    treatment = convert(Array{Float64}, treatment)
    outcome = convert(Array{Float64}, outcome)
    typeof(outcome)




    Array{Float64,1}



We can also index into the data frame using ranges, just like regular matrixes (but we only index on columns). Let's get the baseline covariates. When you get more than one column out of a data frame, you get back another data frame. For some reason that I do not know, `convert` won't work for us here, but `array` will get us what we need (a Julia array instead of a DataFrame).


    baseline_covars = array(df[2:end-1])




    722x7 Array{Real,2}:
     37  11  1  0  1  1     0.0 
     22   9  0  1  0  1     0.0 
     30  12  1  0  0  0     0.0 
     27  11  1  0  0  1     0.0 
     33   8  1  0  0  1     0.0 
     22   9  1  0  0  1     0.0 
     23  12  1  0  0  0     0.0 
     32  11  1  0  0  1     0.0 
     22  16  1  0  0  0     0.0 
     33  12  0  0  1  0     0.0 
     19   9  1  0  0  1     0.0 
     21  13  1  0  0  0     0.0 
     18   8  1  0  0  1     0.0 
      ⋮               ⋮         
     17  10  0  1  0  1     0.0 
     19  11  1  0  0  1     0.0 
     19   9  1  0  0  1  1537.93
     17  10  1  0  0  1     0.0 
     20  11  1  0  0  1  8012.09
     17   8  1  0  0  1   768.81
     18   8  0  0  0  1  1345.91
     20  10  1  0  0  1   825.23
     17   9  0  1  0  1     0.0 
     17   9  1  0  0  1  1206.44
     19  11  1  0  0  1     0.0 
     18  10  1  0  0  1     0.0 



### Formulas

One nice thing about DataFrames.jl is that is has support for formulas. They are similar to R's formulas, but there are some differences. Some packages, like GLM.jl take data frames with formulas as input. Those packages can be used for computing initial estimates, but TargetedLearning.jl does not support formulas and DataFrames directly.  However, you can use DataFrames.jl's functionality to take a DataFrame and a formula and get a numeric design matrix based on a formula.

For example, suppose we'd like to include an age squared term and an interaction term between education and marital status. It looks like polynomial terms aren't implemented currently, so you'll have to manually make those terms.


    #create age squared
    df[:age2] = df[:age] .* df[:age];
    
    #suppress the intercept with -1
    fm = treatment ~ -1 + age + age2 + education + black + hispanic + married + nodegree + RE75 + married&nodegree




    Formula: treatment ~ -1 + age + age2 + education + black + hispanic + married + nodegree + RE75 + married & nodegree




    #take the field named `m` from the created ModelMatrix object
    expanded_baseline_covars = ModelMatrix(ModelFrame(fm, df)).m




    722x9 Array{Float64,2}:
     37.0  1369.0  11.0  1.0  0.0  1.0  1.0     0.0   1.0
     22.0   484.0   9.0  0.0  1.0  0.0  1.0     0.0   0.0
     30.0   900.0  12.0  1.0  0.0  0.0  0.0     0.0   0.0
     27.0   729.0  11.0  1.0  0.0  0.0  1.0     0.0   0.0
     33.0  1089.0   8.0  1.0  0.0  0.0  1.0     0.0   0.0
     22.0   484.0   9.0  1.0  0.0  0.0  1.0     0.0   0.0
     23.0   529.0  12.0  1.0  0.0  0.0  0.0     0.0   0.0
     32.0  1024.0  11.0  1.0  0.0  0.0  1.0     0.0   0.0
     22.0   484.0  16.0  1.0  0.0  0.0  0.0     0.0   0.0
     33.0  1089.0  12.0  0.0  0.0  1.0  0.0     0.0   0.0
     19.0   361.0   9.0  1.0  0.0  0.0  1.0     0.0   0.0
     21.0   441.0  13.0  1.0  0.0  0.0  0.0     0.0   0.0
     18.0   324.0   8.0  1.0  0.0  0.0  1.0     0.0   0.0
      ⋮                            ⋮                     
     17.0   289.0  10.0  0.0  1.0  0.0  1.0     0.0   0.0
     19.0   361.0  11.0  1.0  0.0  0.0  1.0     0.0   0.0
     19.0   361.0   9.0  1.0  0.0  0.0  1.0  1537.93  0.0
     17.0   289.0  10.0  1.0  0.0  0.0  1.0     0.0   0.0
     20.0   400.0  11.0  1.0  0.0  0.0  1.0  8012.09  0.0
     17.0   289.0   8.0  1.0  0.0  0.0  1.0   768.81  0.0
     18.0   324.0   8.0  0.0  0.0  0.0  1.0  1345.91  0.0
     20.0   400.0  10.0  1.0  0.0  0.0  1.0   825.23  0.0
     17.0   289.0   9.0  0.0  1.0  0.0  1.0     0.0   0.0
     17.0   289.0   9.0  1.0  0.0  0.0  1.0  1206.44  0.0
     19.0   361.0  11.0  1.0  0.0  0.0  1.0     0.0   0.0
     18.0   324.0  10.0  1.0  0.0  0.0  1.0     0.0   0.0



It's clunky, but will get the job done. More detailed documentation is [here](http://dataframesjl.readthedocs.org/en/latest/formulas.html).

### Missing data and categorical data

[DataFrames.jl](http://dataframesjl.readthedocs.org/en/latest/) and [DataArrays.jl](https://github.com/JuliaStats/DataArrays.jl) have ways of handling both missing data and categorical data. TargetedLearning.jl does not, so you'll have to deal with those issues ahead of time. The documentation for those packages has more information on both.


    
