#!/bin/bash

cd build
make
str=`find -P jax_plate_lib*`
end_str="../../jax_plate/"$str
cp $str $end_str
