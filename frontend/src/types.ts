export const ANATOMY_OPTIONS = [
  'XR_ELBOW',
  'XR_FINGER',
  'XR_FOREARM',
  'XR_HAND',
  'XR_HUMERUS',
  'XR_SHOULDER',
  'XR_WRIST',
] as const;

export type Anatomy = (typeof ANATOMY_OPTIONS)[number];

export type StudyInput = {
  id: string;
  name: string;
  anatomy: Anatomy;
  files: File[];
};
