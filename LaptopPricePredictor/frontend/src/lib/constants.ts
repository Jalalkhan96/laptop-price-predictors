// Fallback option lists mirroring the backend /metadata response.
// Used when the API is unreachable so the form can still render.
export const FALLBACK_METADATA = {
  brands: [
    'Apple',
    'Dell',
    'HP',
    'Lenovo',
    'Asus',
    'Acer',
    'MSI',
    'Samsung',
    'Huawei',
    'Microsoft',
  ],
  processor_brands: ['Intel', 'AMD', 'Apple Silicon'],
  processor_series: {
    Intel: ['i3', 'i5', 'i7', 'i9'],
    AMD: ['Ryzen 3', 'Ryzen 5', 'Ryzen 7', 'Ryzen 9'],
    'Apple Silicon': ['M1', 'M2', 'M3', 'M4'],
  },
  ram: ['4GB', '8GB', '16GB', '24GB', '32GB', '64GB'],
  storage_types: ['HDD', 'SSD', 'NVMe SSD'],
  storage_sizes: [128, 256, 512, 1024, 2048],
  gpus: [
    'Integrated',
    'Intel Iris Xe',
    'RTX 2050',
    'RTX 3050',
    'RTX 4050',
    'RTX 4060',
    'RTX 4070',
    'RX 7600',
  ],
  display_sizes: [13.0, 14.0, 15.6, 16.0, 17.0],
  resolutions: ['HD', 'FHD', '2K', '4K'],
  refresh_rates: [60, 120, 144, 165, 240],
  operating_systems: ['Windows', 'Linux', 'MacOS', 'DOS'],
  touch_options: ['Yes', 'No'],
  batteries: [40, 56, 70, 90],
};

export const SITE_METADATA = {
  title: 'Smart Laptop Price Predictor',
  tagline: 'AI-powered laptop price prediction using 8 compared regression models.',
  bestR2: '94.2%',
  modelsCompared: '8',
  predictions: '50K+',
};

export const NAV_LINKS = [
  { name: 'Home', href: '/' },
  { name: 'Predict', href: '/predict' },
  { name: 'Model Performance', href: '/model-performance' },
  { name: 'About', href: '/about' },
];
