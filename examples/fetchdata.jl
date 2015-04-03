#Attempt to download the Lalonde data set from http://users.nber.org/~rdehejia/nswdata2.html

function fetchdata(refetch=false)
    datapath = normpath(joinpath(@__FILE__, "..", "data"))
    dataname = joinpath(datapath, "lalonde.csv")

    if isfile(dataname) && !refetch
        info("$dataname exists. To redownload, call fetchdata(true)")
        return
    end

    colnames =["treatment", "age", "education", "black", "hispanic",
               "married", "nodegree", "RE75", "RE78"]

    info("Downloading and parsing http://www.nber.org/~rdehejia/data/nsw_treated.txt...")
    treated  = readdlm(download("http://www.nber.org/~rdehejia/data/nsw_treated.txt"))
    info("Downloading and parsing http://www.nber.org/~rdehejia/data/nsw_control.txt...")
    control = readdlm(download("http://www.nber.org/~rdehejia/data/nsw_control.txt"))
    mat = vcat(reshape(colnames, 1, 9),  treated, control)


    isdir(datapath) || mkdir(datapath)

    info("Writing data set to $dataname")
    writecsv(dataname, mat)
    info("Done!")
end

fetchdata()
