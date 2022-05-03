# DBSCAN Clustering
#
#   References:
#
#       Martin Ester, Hans-peter Kriegel, Jörg S, and Xiaowei Xu
#       A density-based algorithm for discovering clusters
#       in large spatial databases with noise. 1996.
#

using DataStructures
using Plots

mutable struct Object
    Processed::Bool
    reachabilityDistance::Real
    core_distance::Real
    coordinates::Array{Float64}
    label::Int64
end
mutable struct OrderedFile
    objs::Array{Object}
    index::Int
end
function it(file::OrderedFile, new::Object)
    file.objs[file.index] = new
    file.index += 1
    println("****\nIndex: ",file.index)
end

function OPTICS(data, ε::Real, minpts::Int, dist)
    SetOfObjects = Object.(false, NaN, NaN, data)
    OF = OrderedFile(Array{Object}(undef, length(SetOfObjects)), 1)
    for i in 1:length(SetOfObjects)
        Obj = SetOfObjects[i]
        if !Obj.Processed
            ExpandClusterOrder(SetOfObjects, Obj, ε, minpts, dist, OF)
        end
    end
    return OF
end

function ExpandClusterOrder(SetOfObjects, Obj, ε::Real, minpts::Int, dist, OF)
    neighbors = getNeighbors(SetOfObjects, Obj, ε, dist)
    Obj.Processed = true
    Obj.reachabilityDistance = NaN
    setCoreDistance!(Obj, neighbors, ε, minpts)
    it(OF, Obj)
    OrderSeeds = PriorityQueue{Object, Float64}()
    println("core dist: ",Obj.core_distance)

    if isnan(Obj.core_distance)
        update!(OrderSeeds, neighbors, Obj, dist)
        println(length(OrderSeeds), "\n")
        while length(OrderSeeds) > 0
            currentObject = dequeue!(OrderSeeds)
            neighbors = getNeighbors(SetOfObjects, currentObject, ε, dist)
            currentObject.Processed = true
            it(OF, currentObject)
            if isnan(currentObject.core_distance)
               update!(OrderSeeds, neighbors, currentObject, dist) 
            end
        end
    end

end

function update!(OrderSeeds, neighbors, CenterObject, dist)

    c_dist = CenterObject.core_distance
    while length(neighbors)>0
        Obj = dequeue!(neighbors)
        println("neighbor ", Obj)
        if !Obj.Processed
            #println("dist ", c_dist*!isnan(c_dist), " ", dist(CenterObject.coordinates, Obj.coordinates))
            new_r_dist = max(c_dist*!isnan(c_dist), dist(CenterObject.coordinates, Obj.coordinates))
            if isnan(Obj.reachabilityDistance)
                Obj.reachabilityDistance = new_r_dist
                OrderSeeds[Obj] = new_r_dist
            elseif new_r_dist < Obj.reachabilityDistance
                Obj.reachabilityDistance = new_r_dist
                OrderSeeds[Obj] = new_r_dist
            end
            #println(Obj, " ", OrderSeeds[Obj])
        end
    end
end

function getNeighbors(SetOfObjects, Obj, ε, dist)
    output = PriorityQueue{Object, Float64}()
    d = 0
    for potentialNeighbor ∈ SetOfObjects
        if potentialNeighbor != Obj
            d = dist(potentialNeighbor.coordinates, Obj.coordinates)
            if d <= ε
                output[potentialNeighbor] = d
            end
        end
    end
    return output
end

function setCoreDistance!(object, neighbors, ε, minpts)
    if length(neighbors) < minpts
        object.core_distance = NaN
    else
        temp = Array{Pair{Object, Float64}}(undef, minpts-1)
        for i in 1:(minpts-1)
            temp[i] = peek(neighbors)
            dequeue!(neighbors)
        end
        object.core_distance = peek(neighbors).second
        for i in 1:(minpts-1)
            enqueue!(neighbors, temp[i])
        end
    end

    println(object)
    println("CD: ", object.core_distance, "\n")
end



function ExtractClustering(ClusterOrderedObjs, ξ, minpts)

    n = length(ClusterOrderedObjs.objs)

    r = [o.reachabilityDistance for o in ClusterOrderedObjs.objs]
    print(r)
    SetOfSteepDownAreas = []
    SetOfClusters = []
    mib = 0.0
    ξ_compliment = 1-ξ

    ratio = [NaN; r[2:end] ./ r[1:(end-1)]]
    SteepDown = ratio .>= 1/ξ_compliment
    SteepUp = ratio .<= ξ_compliment
    down = ratio .>= 1
    up = ratio .<= 1

    index = 1
    while index < n

        mib = max(mib, r[index])

        if SteepDown[index]
            filterUpdate!(SetOfSteepDownAreas, ξ_compliment, mib, r)

            D_start = index
            # expand down area
            laststeep = index
            index += 1
            nonsteep = 0
            while nonsteep < minpts && index <= n
                if down[index]
                    if !SteepDown[index]
                        nonsteep += 1
                    else
                        laststeep = index
                    end
                    index += 1
                else
                    break
                end
            end
            D_end = laststeep
            index = D_end+1
            push!(SetOfSteepDownAreas, [(D_start, D_end), 0.0])
            # end expand down area
            mib = r[index]
           
        elseif SteepUp[index]
            filterUpdate!(SetOfSteepDownAreas, ξ_compliment, mib, r)
            
            U_start = index
            # expand down area
            laststeep = index
            index += 1
            nonsteep = 0
            while nonsteep < minpts && index <= n
                if up[index]
                    if !SteepUp[index]
                        nonsteep += 1
                    else
                        laststeep = index
                    end
                    index += 1
                else
                    break
                end
            end
            U_end = laststeep
            index = U_end+1
            # end expand down area
            mib = r[index]

            U_clusters = []
            for D in SetOfSteepDownAreas
                c_start = D[1][1]
                c_end = U_end
                # check definition 11
                
                D_max = r[D[1][1]]
                if D_max * ξ_compliment >= r[c_end+1]
                    while r[c_start+1] > r[c_end+1] && c_start < D[1][2]
                        c_start+=1
                    end
                elseif r[c_end+1]*ξ_compliment >= D_max
                    while r[c_end-1] > D_max && c_end > U_start
                        c_end -= 1
                    end
                end

                # criteria 3.a, 1, 2
                if c_end - c_start + 1 >= minpts && c_start <= D[1][2] && c_end >= U_start
                    push!(U_clusters, (c_start, c_end))
                end
            end
            append!(SetOfClusters, reverse(U_clusters))
        else
            index +=1 
        end

    end
    return SetOfClusters
end

function ExtractLabels(Clusters, OF)
    labels = Array{Int64}(-1, length(OF.objs))
    label = 0
    for c in Clusters
        if sum(labels[c[1]:(c[2]+1)] != -1)>0
            labels[c[1]:(c[2]+1)] = label
            label +=1
        end
    end

    for i in 1:length(labels)
        OF.objs[i].label = labels[i]
    end
end

function filterUpdate!(SSDA, ξ_compliment, mib, r)
    if isnan(mib)
        SSDA = []
    end
    if length(SSDA)>0
        SSDA = SSDA[[mib <= r[sda[1][1]] * ξ_compliment for sda in SSDA]]
    end
    for sda in SSDA
        println(sda, " ", mib)
        sda[2] = max(sda[2], mib)
    end
end

function regDist(a, b)
    return sqrt(sum((a-b).^2))
end

data = [[[4, -1]+0.01*randn(2) for i in 1:250]; [[1, -2] + 0.02 *randn(2) for i in 1:250]]
of = OPTICS(data, 0.005, 100, regDist)
clusterings = ExtractClustering(of, .75, 100)
r = [o.reachabilityDistance for o in of.objs]
c = [o.core_distance for o in of.objs]
x = [o.coordinates[1] for o in of.objs]
y = [o.coordinates[2] for o in of.objs]

scatter(x, y)


scatter(r)

