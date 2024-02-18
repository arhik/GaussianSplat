using Rotations
using LinearAlgebra
using Quaternions
using CoordinateTransformations
using StaticArrays
using JSON

coordinateTransform = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1] .|> Float32

mutable struct Camera
    fx
    fy
    far
    near
    eye
    lookat
    up
    scale
    aspectRatio
    id
end

function defaultCamera(;id=0)
	eye = [0.0, 0.0, 50.0] .|> Float32
	lookat = [0, 0, 0] .|> Float32
	up = [0, 1, 0] .|> Float32
	scale = [1, 1, 1] .|> Float32
    fx = 100.0f0
    fy = 100.0f0
	aspectRatio = 1.0 |> Float32
	nearPlane = 0.1 |> Float32
	farPlane = 100.0 |> Float32
	return Camera(
        fx,
        fy,
        farPlane,
        nearPlane,
		eye,
		lookat,
		up,
		scale,
		aspectRatio,
		id
	)
end

function rotateTransform(q::Quaternion)
	rotMat = coordinateTransform[1:3, 1:3]*rotmatrix_from_quat(q)
	mat = Matrix{Float32}(I, (4, 4))
	mat[1:3, 1:3] .= rotMat
	return LinearMap(
		SMatrix{4, 4}(
			mat
		)
	)
end

function scaleTransform(loc)
	(x, y, z) = coordinateTransform[1:3, 1:3]*loc
	return LinearMap(
		@SMatrix(
			[
				x   0   0   0;
				0   y   0   0;
				0   0   z   0;
				0   0   0   1;
			]
		) .|> Float32
	)
end

function translateCamera(camera::Camera)
	(x, y, z) = coordinateTransform[1:3, 1:3]*camera.eye
	return LinearMap(
		@SMatrix(
			[
				1 	0 	0 	x;
				0 	1 	0 	y;
				0 	0 	1 	z;
				0 	0 	0 	1;
			]
		) .|> Float32
	) |> inv
end

function computeTransform(camera::Camera)
	eye = camera.eye
	lookat = camera.lookat
	up = camera.up
	w = -(lookat .- eye) |> normalize
	u =	cross(up, w) |> normalize
	v = cross(w, u)
	m = MMatrix{4, 4, Float32}(I)
	m[1:3, 1:3] .= (cat([u, v, w]..., dims=2) |> adjoint .|> Float32 |> collect)
	m = SMatrix(m)
	return LinearMap(m) ∘ translateCamera(camera)
end

function computeProjection(camera::Camera, w, h)
    p = MArray{Tuple{4, 4}, Float32}(undef)
    p .= 0.0f0
    p[1, 1] = 2.0f0*(camera.fx)/w
    p[2, 2] = 2.0f0*(camera.fy)/h
    p[3, 3] = (camera.far + camera.near)/(camera.far - camera.near)
    p[3, 4] = -2.0f0*(camera.far*camera.near)/(camera.far - camera.near)
    p[4, 3] = 1
    return LinearMap(p)
end

function loadCameras(path)
	cameras = JSON.parsefile(path)
	return cameras
end

struct GroundTruthCamera
	position
	rotation
	fx
	fy
	width
	height
	id
	imgName
end

function getCamera(path, idx)
	cameras = loadCameras(path)
	camera = cameras[idx]
	position = camera["position"]
	rotation = cat(camera["rotation"]..., dims=2)
	fx = camera["fx"]
	fy = camera["fy"]
	width = camera["width"]
	height = camera["height"]
	imgName = camera["img_name"]
	id = camera["id"]
	return GroundTruthCamera(
		position,
		rotation,
		fx,
		fy,
		width,
		height,
		id,
		imgName
	)
end


function computeTransform(camera::GroundTruthCamera)
	m = MMatrix{4, 4, Float32}(I)
	m[1:3, 1:3] .= -adjoint(camera.rotation)
	m[1:3, 4] .= camera.position
	return LinearMap(m)
end


function computeProjection(camera::GroundTruthCamera, near, far)
    p = MArray{Tuple{4, 4}, Float32}(undef)
    p .= 0.0f0
    p[1, 1] = 2.0f0*(camera.fx)/camera.width
    p[2, 2] = 2.0f0*(camera.fy)/camera.height
    p[3, 3] = (far + near)/(far - near)
    p[3, 4] = -2.0f0*(far*near)/(far - near)
    p[4, 3] = 1
    return LinearMap(p)
end
