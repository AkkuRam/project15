%%
clear; close all; clc;
%% standard parameters
g = 9.81;
depth = 30;
gamma = 3.3;
gammacoastal = 2;
Tp = 8.533334;
fp = 1/Tp;
Hs = 2.512497;

sigma1 = 0.07;
sigma2 = 0.09;
sigma = @(f) (f <= fp) .* sigma1 + (f > fp) .* sigma2;

a = 0.0081;

f = linspace(0.01, 5*fp, 1000); % Frequency range for the spectrum

%% Create spetrum
for j = 1:3
    Sjon = @(f) a * g^2 * (2*pi)^(-4) .* f.^(-5) .* exp(-(5/4)*(fp./f).^4).* gamma.^(exp(-( (f./fp - 1).^2 ) ./ (2*(sigma(f)).^2)));
    m0 = integral(Sjon, 0, 5*fp);
    Hs_computed = 4*sqrt(m0);
    a = a * (Hs / Hs_computed)^2;
end
Sjon = @(f) a * g^2 * (2*pi)^(-4) .* f.^(-5) .* exp(-(5/4)*(fp./f).^4).* gamma.^(exp(-( (f./fp - 1).^2 ) ./ (2*(sigma(f)).^2)));
Sjon_coastal = @(f) a * g^2 * (2*pi)^(-4) .* f.^(-5) .* exp(-(5/4)*(fp./f).^4).* gammacoastal.^(exp(-( (f./fp - 1).^2 ) ./ (2*(sigma(f)).^2))); % this Sjon with coastal gamma


%% Create coastal spectrum

omega = 2*pi*f;
k = zeros(size(f));
for i = 1:length(f)
    k_guess = omega(i)^2 / g;
    k(i) = fzero(@(kk) g*kk*tanh(kk*depth) - omega(i)^2, k_guess);
end

% Coastal spectrum
tanh_kh2 = tanh(k.*depth).^2;
a = 1;

% Match the area under the curves to the real area under the curve
for j = 1:3
    S_coastal = Sjon_coastal(f) .* tanh_kh2 .* a;
    m0 = trapz(f, S_coastal);      
    Hs_computed = 4*sqrt(m0);
    a = a * (Hs / Hs_computed)^2;
end

S_coastal = @(f) Sjon_coastal(f) .* tanh_kh2 .* a;


%% Plot the spectrum
figure;
plot(f, Sjon(f), 'b', 'LineWidth', 2);   % deep-water JONSWAP
hold on;                                 % keep the figure for additional plots
plot(f, S_coastal(f), 'r', 'LineWidth', 2); % coastal JONSWAP (vector, not function handle)
grid on;

xlabel('Frequency [Hz]');
ylabel('S(f) [m^2/Hz]');
title('Deep Water vs Coastal JONSWAP Spectrum');

legend('Deep-water JONSWAP', 'Coastal JONSWAP'); % add legend

%% Preallocate deepwater
duration = 5; dt = 0.03; t = 0:dt:duration; % time vector
x = linspace(0,2.5,100);                    % spatial grid
df = f(2) - f(1);

% Component amplitudes and phases
ampl = sqrt(2 * Sjon(f) * df);
phi = 2*pi*rand(size(f));

% Wavenumbers (deep water)
omega = 2*pi*f;       % angular frequency
k = omega.^2 / g;     


%% animate 1D without direction deepwater
figure;
h = plot(x, zeros(size(x)));
ylim([-2*Hs 2*Hs]);    % use Hs scaling
xlabel('x'); ylabel('Surface elevation \eta (m)');
title('JONSWAP Irregular Sea');

eta_time_series = zeros(length(t),1);
x0 = 1;

for it = 1:length(t)
    % Surface elevation eta(x,t): sum of sinusoids
    Ysum = zeros(size(x));
    for j = 1:length(f)
        Ysum = Ysum + ampl(j)*cos(k(j)*x - omega(j)*t(it) + phi(j));
    end
    eta_time_series(it) = Ysum(x0);
    set(h,'YData',Ysum);
    title(sprintf('t = %.2f s',t(it)));
    drawnow;
end

%% Check Hs 
Hs_est = 4*std(eta_time_series);
fprintf('Hs (theory from PM) = %.3f m\n', Hs);
fprintf('Hs (estimate from realization) = %.3f m\n', Hs_est);


% Check if they differ by more than 10%
if abs(Hs - Hs_est)/Hs > 0.10
    warning('Hs estimate differs from theoretical value by more than 10%%!');
end

%% Generate directional spreading model based on cosine 2s
s = 10;
thetap = 15.476746;
k = 1;

% Normalize
D = @(theta) k * cos((theta - thetap)/2).^(2*s);
norm = integral(D, 0, 2*pi);
k = k/norm;

% Combine JONSWAP with directional spreading
D = @(theta) k * cos((theta - thetap)/2).^(2*s);
S_directional = @(f, theta) Sjon(f) .* D(theta);

%% Plot two dimensional spectrum'

f_range = linspace(0.01, 5*fp, 200);
theta_range = linspace(0, 2*pi, 200);
[F, THETA] = meshgrid(f_range, theta_range);

S_dir = S_directional(F, THETA);

figure;
surf(THETA, F, S_dir); % requires polarPcolor function (see below)
title('2D Directional JONSWAP Spectrum');
colormap turbo;
colorbar;

%% Generate amplitudes and random phi's

df = f_range(2) - f_range(1);
d_theta = theta_range(2) - theta_range(1);

Ampl = @(f,theta) sqrt(2 * S_directional(f,theta) * df * d_theta);

Ampl_matrix = zeros(length(theta_range), length(f_range));

for i = 1:length(theta_range)
    for j = 1:length(f_range)
        Ampl_matrix(i, j) = Ampl(i,j);
    end
end


Phi_matrix = 2 * pi * rand(size(Ampl_matrix));

%% Solve dispersion
Nf = numel(f);

Nt = 2000;                    
t_end = 200;                  
t = linspace(0, t_end, Nt);   
dft = t(2)-t(1);

% --- dispersion solver: solve for k for each frequency ---
% solve omega^2 = g*k * tanh(k*h)
k_disp = zeros(size(omega));
for ni = 1:Nf
    wn = omega(ni);
    % initial guess: deep water k = wn^2/g
    k0 = wn^2 / g;
    kguess = k0;
    for it = 1:30
        fdisp = g*kguess.*tanh(kguess*depth) - wn^2;
        dfdisp = g*tanh(kguess*depth) + g*kguess.*depth .* (1./cosh(kguess*depth)).^2;
        knew = kguess - fdisp./dfdisp;
        if abs(knew - kguess) < 1e-10
            break
        end
        kguess = knew;
    end
    k_disp(ni) = kguess;
end
%% Compute wave-induced surface elevation and pressures on panels

Nx = 100; Ny = 100;
x = linspace(-250, 250, Nx);
y = linspace(-250, 250, Ny);
[X, Y] = meshgrid(x, y);

Nt = 200; 
t_end = 20; 
t = linspace(0, t_end, Nt);

eta = gpuArray.zeros(Ny, Nx, Nt);

% Expand spatial and time grids
Xg = reshape(X, [Ny, Nx, 1]);
Yg = reshape(Y, [Ny, Nx, 1]);
tvec = reshape(t, [1, 1, Nt]);

for ni = 1:length(f_range)         % loop over frequencies (200)
    omega_n = 2*pi*f_range(ni);
    k_n = k_disp(ni);
    
    for mi = 1:length(theta_range) % loop over directions (200)
        theta_m = theta_range(mi);
        A_nm = Ampl(f_range(ni), theta_range(mi));
        phi_nm = Phi_matrix(mi, ni);

        % Spatial phase (vectorized in space)
        phase_spatial = k_n*(Xg*cos(theta_m) + Yg*sin(theta_m)) + phi_nm;

        % Contribution over time (vectorized)
        eta = eta + A_nm * cos(phase_spatial - omega_n * tvec);
    end
end

eta_cpu = gather(eta);



%% Visualize 3D

figure;

% Precompute axis limits
x_min = min(X(:));
x_max = max(X(:));
y_min = min(Y(:));
y_max = max(Y(:));
z_min = -Hs;
z_max = Hs;

% Create surf plot once
hSurf = surf(X, Y, eta_cpu(:,:,1), 'EdgeColor', 'none');
colormap(turbo);
colorbar;
caxis([z_min z_max]);
axis([x_min x_max y_min y_max z_min z_max]);
xlabel('x (m)');
ylabel('y (m)');
zlabel('\eta (m)');
view(45,30);
shading interp; % optional: smoother surface

for ti = 1:Nt
    hSurf.ZData = eta_cpu(:,:,ti);
    title(sprintf('Time = %.2f s', t(ti)));
    drawnow;
    pause(0.01);  % pauses 0.1 seconds between frames
end

%% Compute wave-induced surface elevation and pressures on panels

% Panel/grid parameters
Nx = 30;
Ny = 30;
x = linspace(-50, 50, Nx);
y = linspace(-50, 50, Ny);
[X, Y] = meshgrid(x, y);

% Time parameters
Nt = 100;
t_end = 10;
t = linspace(0, t_end, Nt);

% Transfer variables to GPU
X = gpuArray(X);
Y = gpuArray(Y);
t = gpuArray(t);
f_range = gpuArray(f_range);
theta_range = gpuArray(theta_range);
Ampl_matrix = gpuArray(Ampl_matrix);
Phi = gpuArray(Phi_matrix);

% Initialize surface elevation
eta = gpuArray.zeros(Ny, Nx, Nt);

% Loop over frequencies and directions
for ni = 1:length(f_range)
    omega_n = 2 * pi * f_range(ni);
    k_n = k_disp(ni);  % Assume k_disp returns wavenumber for f_n
    
    for mi = 1:length(theta_range)
        theta_m = theta_range(mi);
        A_nm = Ampl(f_range(ni), theta_range(mi));
        phi_nm = Phi_matrix(mi, ni);
        
        % Spatial phase
        kx = k_n * cos(theta_m);
        ky = k_n * sin(theta_m);
        phase_spatial = kx * X + ky * Y + phi_nm;
        
        % Time evolution
        for ti = 1:Nt
            eta(:, :, ti) = eta(:, :, ti) + A_nm * cos(phase_spatial - omega_n * t(ti));
        end
    end
end

% Gather results back to CPU
eta_cpu = gather(eta);
