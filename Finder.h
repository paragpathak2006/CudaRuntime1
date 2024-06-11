#pragma once
#include <tuple>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "Point.h"
typedef long long int Lint;
typedef Lint Key;
typedef pair<int, int> Node_Pair;
typedef vector<Lint> Octet;

// Point Hash functor
struct Hash_of_Point {
    static double map_size;
    size_t operator() (const Point& P) const {
        int Px = round(P.x / map_size);
        int Py = round(P.y / map_size);
        int Pz = round(P.z / map_size);
        return (Px * 18397) + (Py * 20483) + (Pz * 29303);
    }
};

// Point Equal functor
struct Equal_of_Point {
    static double map_size;
    bool operator()(const Point& P1, const Point& P2) const {
        if (round(P1.x / map_size) != round(P2.x / map_size)) return false;
        if (round(P1.y / map_size) != round(P2.y / map_size)) return false;
        if (round(P1.z / map_size) != round(P2.z / map_size)) return false;
        return true;
    }
};


typedef unordered_multimap< Point, int, Hash_of_Point, Equal_of_Point> Point_Map;

#define for_range(it,range) for(auto& it = range.first; it != range.second; ++it)
class Space_map {
public:
    static Point_Map point_map;
    static double map_size;
    static double beta;

    static void add_point(const Point& P, int index) { point_map.emplace(P, index); }

    static void set_map_size(double mapsize) {
        map_size = mapsize; 
        Equal_of_Point::map_size = mapsize;
        Hash_of_Point::map_size = mapsize;
    }

    static void initialize_space_map(const Points &points, double mapsize, double beta_) {
        //FOR(i, j, k) octet(i, j, k) = hasher(i-1, j - 1, k-1);
        set_map_size(mapsize);
        beta = beta_;
        int num_points = points.size();
        point_map.reserve(num_points * 2);

        for (size_t i = 0; i < num_points; i++)
            add_point(points[i], i);

//        cout << "Point map print" << endl;
//        for (const auto& p : point_map)
//            p.first.print();
//        cout << endl << endl;
//        vector<int> point_indexes;
//        lookup_point_region(Point(0, 0, 0.02), point_indexes);
    }

    static void make_empty() {point_map.empty();}

    static Point neareast_map_point(const Point& P, int i, int j, int k) {
        return Point(P.x + i * Space_map::map_size, P.y + j * Space_map::map_size, P.z + k * Space_map::map_size);
    }

    static void lookup_point_region(const Point& P, vector<int>& point_indexes) {
        //auto hash_P = hasher(P);
        //auto hash_value = hash_P + octet(i, j, k);//Using memonization
        auto count = point_map.count(P);
//        cout << "Lookup nearby point at dist : " << map_size << " , Point : "; P.print();

        if (count > 0) {
            auto range = point_map.equal_range(P);

            cout << "Points found" << endl;
            for_range(point, range) {
                int point_index = point->second;
                point_indexes.push_back(point_index);
                point->first.print();
            }
        }
    }


    static void lookup_region(const Point& target, vector<int>& point_indexes) {
        int max_index = round(beta / Space_map::map_size);
        point_indexes.reserve(50);
        //auto hash_P = hasher(P);
        cout << "Lookup nearby point at box dist : " << map_size * max_index<<endl;
        cout << "Box dim : " << endl;
        neareast_map_point(target, max_index, max_index, max_index).print();
        neareast_map_point(target, -max_index, -max_index, -max_index).print();
        cout << endl << endl;


        for (int i = -max_index; i <= max_index; i++)
            for (int j = -max_index; j <= max_index; j++)
                for (int k = -max_index; k <= max_index; k++)
                    lookup_point_region(neareast_map_point(target, i, j, k), point_indexes);
    }

    static double search_space_map(const Points& points, const Point& target, int &nearest_point) {
        vector<int> point_indexes;
        double beta2 = beta * beta;
        lookup_region(target, point_indexes);


        double min_dist = target.dist(points[0]);

        for (int i : point_indexes)
        {
            float dist = target.dist(points[i]);
            if (dist < min_dist && dist < beta2)
            {
                min_dist = dist;
                nearest_point = i;
            }
        }
        return (min_dist > beta2) ? beta2 : min_dist;
    }

};

/* Edge Hash functor
struct Hash_of_Edge {
    size_t operator() (const Edge& P) const {
        auto n0 = P.node[0];
        auto n1 = P.node[1];
        if (n0 < n1) return n0 * 18397 + n1 * 20483;
        return n1 * 18397 + n0 * 20483;
    }
};

// Edge Equal functor
struct Equal_of_Edge {
    bool operator()(const Edge& P1, const Edge& P2) const {
        auto a0 = P1.node[0];        auto a1 = P1.node[1];
        auto b0 = P2.node[0];        auto b1 = P2.node[1];

        if (a0 == b0 && a1 == b1) return true;
        if (a1 == b0 && a0 == b1) return true;

        return false;
    }
};
*/

//typedef unordered_map< Point, Points> Point_Hash;
//typedef unordered_map< Point, Nodes > Node_Hash;

