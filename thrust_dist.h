#pragma once

struct dist_sqxy
{
    const double x, y;
    dist_sqxy(double _x, double _y) : x(_x), y(_y) {}
    __host__ __device__
        double operator()(const double& X, const double& Y) const {
        return (X - x) * (X - x) + (Y - y) * (Y - y);
    }
};

struct dist_sqz
{
    const double z;
    dist_sqz(double _z) : z(_z) {}
    __host__ __device__
        double operator()(const double& Z, const double& Y) const {
        return (Z - z) * (Z - z) + Y;
    }
};

struct mim_dist
{
    __host__ __device__
        double operator()(const double& Z, const double& Y) const {
        return (Z < Y) ? Z : Y;
    }
};
