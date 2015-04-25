#Attempt to download the Lalonde data set from http://users.nber.org/~rdehejia/nswdata2.html

function fetchdata(;refetch=false, ds = "lalonde_dw")
    if ds == "lalonde"
        csvname = "lalonde.csv"
        fname = "nsw"
        colnames =["treatment", "age", "education", "black", "hispanic",
                   "married", "nodegree", "RE75", "RE78"]
    elseif ds == "lalonde_dw"
        csvname = "lalonde_dw.csv"
        fname = "nswre74"
        colnames = ["treatment", "age", "education", "black", "hispanic",
                    "married", "nodegree", "RE74", "RE75", "RE78"]
    else
        error("unknown ds")
    end

    datapath = normpath(joinpath(@__FILE__, "..", "data"))
    dataname = joinpath(datapath, csvname)

    if isfile(dataname) && !refetch
        info("$dataname exists. To redownload, call fetchdata(ds=\"$ds\", refetch=true)")
        return
    end

    info("Downloading and parsing http://www.nber.org/~rdehejia/data/$(fname)_treated.txt...")
    treated  = readdlm(download("http://www.nber.org/~rdehejia/data/$(fname)_treated.txt"))
    info("Downloading and parsing http://www.nber.org/~rdehejia/data/$(fname)_control.txt...")
    control = readdlm(download("http://www.nber.org/~rdehejia/data/$(fname)_control.txt"))
    mat = vcat(reshape(colnames, 1, length(colnames)),  treated, control)


    isdir(datapath) || mkdir(datapath)

    info("Writing data set to $dataname")
    writecsv(dataname, mat)
    info("Done!")
end
