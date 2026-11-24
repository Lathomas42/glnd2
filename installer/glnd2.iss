; Windows installer (Inno Setup 6.3+), built by .github/workflows/build.yml
; after PyInstaller has produced dist\glnd2:
;   iscc /DAppVersion=0.2.0 installer\glnd2.iss  ->  glnd2-windows-x64-setup.exe
#ifndef AppVersion
  #define AppVersion "dev"
#endif

[Setup]
; Fixed forever: identifies glnd2 so new versions upgrade the old install.
AppId={{4D5D227E-C777-4F30-B52E-ACEB8786E644}
AppName=glnd2
AppVersion={#AppVersion}
AppPublisherURL=https://github.com/Lathomas42/glnd2
DefaultDirName={autopf}\glnd2
DisableProgramGroupPage=yes
; Installs for the current user without admin rights (lab PCs); someone with
; admin rights can pick "all users" instead.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..
OutputBaseFilename=glnd2-windows-x64-setup
SetupIconFile=..\icons\glnd2.ico
UninstallDisplayIcon={app}\glnd2.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; Upgrades replace the bundled libraries wholesale, so none from an older
; version are left behind to be loaded by mistake.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "..\dist\glnd2\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\glnd2"; Filename: "{app}\glnd2.exe"
Name: "{autodesktop}\glnd2"; Filename: "{app}\glnd2.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\glnd2.exe"; Description: "{cm:LaunchProgram,glnd2}"; Flags: nowait postinstall skipifsilent
